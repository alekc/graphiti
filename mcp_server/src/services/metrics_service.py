"""Prometheus metrics for the graphiti MCP server (ALE-51).

Three kinds of series, refreshed on different schedules:

- Queue depth: live in-process state (`asyncio.Queue.qsize()`), read
  synchronously on every `/metrics` scrape. Cheap, no I/O.
- Episode processing duration: recorded by queue_service.py as each
  episode/batch actually finishes (or fails) processing. Event-driven,
  not polled.
- Graph-shape metrics: derived from Cypher queries against Neo4j
  (episode/entity/edge counts, ingest lag, empty-content episodes,
  orphaned edges, duplicate episode names). Recomputed on a background
  timer rather than per-scrape, so a slow or frequent scrape never blocks
  on a graph query, and the query cost does not scale with scrape rate.
"""

import asyncio
import logging
import time
from typing import Any

from graphiti_core.driver.driver import GraphProvider
from prometheus_client import CONTENT_TYPE_LATEST, Gauge, Histogram, generate_latest

logger = logging.getLogger(__name__)

QUEUE_DEPTH = Gauge(
    'graphiti_queue_depth',
    'Episodes waiting in the per-group processing queue',
    ['group_id'],
)
EPISODE_PROCESSING_DURATION_SECONDS = Histogram(
    'graphiti_episode_processing_duration_seconds',
    'Wall-clock time to process one episode (single) or one batch (bulk), '
    'from dequeue to completion or failure',
    ['group_id', 'kind', 'status'],
    # Observed range in practice: single episodes ~2.5-3.2 minutes (ALE-49),
    # bulk batches longer. Buckets span seconds to ~20 minutes.
    buckets=(1, 5, 15, 30, 60, 120, 180, 300, 600, 900, 1200),
)
EPISODES_TOTAL = Gauge(
    'graphiti_episodes_total',
    'Total Episodic nodes in the graph',
    ['group_id'],
)
ENTITY_NODES_TOTAL = Gauge(
    'graphiti_entity_nodes_total',
    'Total Entity nodes in the graph',
    ['group_id'],
)
EDGES_TOTAL = Gauge(
    'graphiti_edges_total',
    'Total Entity-to-Entity (RELATES_TO) edges in the graph',
    ['group_id'],
)
EPISODES_EMPTY_CONTENT = Gauge(
    'graphiti_episodes_empty_content',
    'Episodic nodes with empty or null content',
    ['group_id'],
)
EDGES_ORPHANED = Gauge(
    'graphiti_edges_orphaned',
    'Edges whose every source episode has been deleted',
    ['group_id'],
)
EPISODES_DUPLICATE_NAMES = Gauge(
    'graphiti_episodes_duplicate_names',
    'Episodic nodes sharing the same group, name and content as another episode',
    ['group_id'],
)
INGEST_LAG_SECONDS = Gauge(
    'graphiti_ingest_lag_seconds',
    'Seconds since the most recently created (ingest time) episode, per group',
    ['group_id'],
)
LAST_REFRESH_SUCCESS = Gauge(
    'graphiti_metrics_last_refresh_success',
    '1 if the last graph-shape metrics refresh succeeded, 0 otherwise',
)

_EPISODE_STATS_QUERY = """
MATCH (e:Episodic)
RETURN e.group_id AS group_id,
       count(e) AS episodes_total,
       sum(CASE WHEN e.content IS NULL OR e.content = '' THEN 1 ELSE 0 END)
         AS empty_content,
       max(e.created_at.epochSeconds) AS newest_created_at_epoch
"""

_ENTITY_COUNT_QUERY = """
MATCH (n:Entity)
RETURN n.group_id AS group_id, count(n) AS entity_nodes_total
"""

_EDGE_COUNT_QUERY = """
MATCH (:Entity)-[e:RELATES_TO]->(:Entity)
RETURN e.group_id AS group_id, count(e) AS edges_total
"""

_DUPLICATE_EPISODES_QUERY = """
MATCH (e:Episodic)
WITH e.group_id AS group_id, e.name AS name, e.content AS content, count(e) AS c
WHERE c > 1
RETURN group_id, sum(c) AS duplicate_episodes
"""

# An edge is orphaned when none of the episode uuids in its provenance
# list resolve to a live Episodic node. episodes[] is provenance history,
# not a live-source list (a deleted episode's uuid is never removed from
# it), so every uuid has to be resolved against the current graph rather
# than trusting the array's mere presence. See ALE-51 / tool-routing.md.
_ORPHANED_EDGES_QUERY = """
MATCH (:Entity)-[e:RELATES_TO]->(:Entity)
WHERE size(e.episodes) > 0
UNWIND e.episodes AS ep_uuid
OPTIONAL MATCH (ep:Episodic {uuid: ep_uuid})
WITH e.uuid AS edge_uuid, e.group_id AS group_id, count(ep) AS live_sources
WHERE live_sources = 0
RETURN group_id, count(edge_uuid) AS orphaned_edges
"""


def refresh_queue_metrics(queue_service: Any) -> None:
    """Set graphiti_queue_depth from live in-process queue state."""
    for group_id in queue_service.get_known_group_ids():
        QUEUE_DEPTH.labels(group_id=group_id).set(queue_service.get_queue_size(group_id))


def record_episode_processing_duration(
    group_id: str, kind: str, status: str, duration_seconds: float
) -> None:
    """Record one completed (or failed) episode/batch processing duration.

    kind: 'single' (add_episode) or 'bulk' (add_episode_bulk).
    status: 'success' or 'failure'. Recorded either way: a failed call still
    consumed real wall-clock time, and a shift in failure-path duration is
    itself a useful signal (e.g. retries piling up before the raise).
    """
    EPISODE_PROCESSING_DURATION_SECONDS.labels(
        group_id=group_id, kind=kind, status=status
    ).observe(duration_seconds)


async def _run_group_query(driver: Any, query: str) -> list[dict[str, Any]]:
    records, _, _ = await driver.execute_query(query)
    return [dict(record) for record in records]


async def refresh_graph_metrics(driver: Any, now_epoch: float) -> None:
    """Recompute every Cypher-derived gauge. Each query fails independently
    so one bad query does not blank out the others."""
    queries = {
        'episode stats': (_EPISODE_STATS_QUERY, _apply_episode_stats),
        'entity counts': (_ENTITY_COUNT_QUERY, _apply_entity_counts),
        'edge counts': (_EDGE_COUNT_QUERY, _apply_edge_counts),
        'duplicate episodes': (_DUPLICATE_EPISODES_QUERY, _apply_duplicate_episodes),
        'orphaned edges': (_ORPHANED_EDGES_QUERY, _apply_orphaned_edges),
    }
    all_ok = True
    for label, (query, apply_fn) in queries.items():
        try:
            records = await _run_group_query(driver, query)
            apply_fn(records, now_epoch)
        except Exception:
            all_ok = False
            logger.exception('graphiti metrics: %s query failed', label)
    LAST_REFRESH_SUCCESS.set(1 if all_ok else 0)


def _apply_episode_stats(records: list[dict[str, Any]], now_epoch: float) -> None:
    for row in records:
        group_id = row['group_id']
        EPISODES_TOTAL.labels(group_id=group_id).set(row['episodes_total'])
        EPISODES_EMPTY_CONTENT.labels(group_id=group_id).set(row['empty_content'])
        newest = row.get('newest_created_at_epoch')
        if newest is not None:
            INGEST_LAG_SECONDS.labels(group_id=group_id).set(max(now_epoch - newest, 0))


def _apply_entity_counts(records: list[dict[str, Any]], now_epoch: float) -> None:
    del now_epoch
    for row in records:
        ENTITY_NODES_TOTAL.labels(group_id=row['group_id']).set(row['entity_nodes_total'])


def _apply_edge_counts(records: list[dict[str, Any]], now_epoch: float) -> None:
    del now_epoch
    for row in records:
        EDGES_TOTAL.labels(group_id=row['group_id']).set(row['edges_total'])


def _apply_duplicate_episodes(records: list[dict[str, Any]], now_epoch: float) -> None:
    del now_epoch
    for row in records:
        EPISODES_DUPLICATE_NAMES.labels(group_id=row['group_id']).set(row['duplicate_episodes'])


def _apply_orphaned_edges(records: list[dict[str, Any]], now_epoch: float) -> None:
    del now_epoch
    for row in records:
        EDGES_ORPHANED.labels(group_id=row['group_id']).set(row['orphaned_edges'])


async def metrics_refresh_loop(driver: Any, interval_seconds: int = 60) -> None:
    """Background task: recompute graph-shape metrics on a fixed interval.

    No-ops (logs once, exits) for a non-Neo4j driver: the Cypher above is
    Neo4j-specific, and queue-depth metrics do not depend on this loop.
    """
    if getattr(driver, 'provider', None) != GraphProvider.NEO4J:
        logger.warning(
            'graphiti metrics: graph-shape metrics need a Neo4j driver, got %s; '
            'skipping the refresh loop (queue depth metrics are unaffected)',
            getattr(driver, 'provider', type(driver)),
        )
        return

    while True:
        try:
            await refresh_graph_metrics(driver, now_epoch=time.time())
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception('graphiti metrics: refresh loop iteration failed')
        await asyncio.sleep(interval_seconds)


def render_metrics() -> tuple[bytes, str]:
    """Return (body, content_type) for the /metrics HTTP response."""
    return generate_latest(), CONTENT_TYPE_LATEST
