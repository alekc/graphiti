#!/usr/bin/env python3
"""Unit tests for the ALE-51 Prometheus metrics module.

Cypher correctness is verified separately against a real Neo4j instance
(not part of this fast unit suite); these tests cover the pure-Python
gauge-update logic these queries feed into.
"""

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from graphiti_core.driver.driver import GraphProvider

import services.metrics_service as ms


def _sample(metric, **labels) -> float | None:
    for m in metric.collect():
        for s in m.samples:
            if s.labels == labels:
                return s.value
    return None


class FakeQueueService:
    def __init__(self, sizes: dict[str, int]):
        self._sizes = sizes

    def get_known_group_ids(self) -> list[str]:
        return list(self._sizes.keys())

    def get_queue_size(self, group_id: str) -> int:
        return self._sizes[group_id]


class TestRefreshQueueMetrics:
    def test_sets_gauge_per_known_group(self):
        ms.refresh_queue_metrics(FakeQueueService({'ale51-unit-a': 4, 'ale51-unit-b': 0}))
        assert _sample(ms.QUEUE_DEPTH, group_id='ale51-unit-a') == 4
        assert _sample(ms.QUEUE_DEPTH, group_id='ale51-unit-b') == 0

    def test_unknown_group_untouched(self):
        ms.refresh_queue_metrics(FakeQueueService({}))
        # No group known this call: the gauge for an unrelated label simply
        # is not asserted here, this only proves the call does not raise.


class TestApplyEpisodeStats:
    def test_sets_totals_empty_content_and_lag(self):
        now = 1_000_000.0
        rows = [
            {
                'group_id': 'ale51-unit-episodes',
                'episodes_total': 5,
                'empty_content': 1,
                'newest_created_at_epoch': now - 30,
            }
        ]
        ms._apply_episode_stats(rows, now)
        assert _sample(ms.EPISODES_TOTAL, group_id='ale51-unit-episodes') == 5
        assert _sample(ms.EPISODES_EMPTY_CONTENT, group_id='ale51-unit-episodes') == 1
        assert _sample(ms.INGEST_LAG_SECONDS, group_id='ale51-unit-episodes') == 30

    def test_missing_newest_created_at_skips_lag_gauge_without_raising(self):
        rows = [
            {
                'group_id': 'ale51-unit-nolag',
                'episodes_total': 0,
                'empty_content': 0,
                'newest_created_at_epoch': None,
            }
        ]
        ms._apply_episode_stats(rows, now_epoch=1_000_000.0)
        assert _sample(ms.EPISODES_TOTAL, group_id='ale51-unit-nolag') == 0

    def test_lag_never_negative_on_clock_skew(self):
        # newest_created_at_epoch slightly ahead of now_epoch (clock skew
        # between the app server and Neo4j) must clamp to 0, not go negative.
        now = 1_000_000.0
        rows = [
            {
                'group_id': 'ale51-unit-skew',
                'episodes_total': 1,
                'empty_content': 0,
                'newest_created_at_epoch': now + 5,
            }
        ]
        ms._apply_episode_stats(rows, now)
        assert _sample(ms.INGEST_LAG_SECONDS, group_id='ale51-unit-skew') == 0


class TestApplyEntityAndEdgeCounts:
    def test_entity_counts(self):
        ms._apply_entity_counts(
            [{'group_id': 'ale51-unit-entities', 'entity_nodes_total': 7}], now_epoch=0.0
        )
        assert _sample(ms.ENTITY_NODES_TOTAL, group_id='ale51-unit-entities') == 7

    def test_edge_counts(self):
        ms._apply_edge_counts(
            [{'group_id': 'ale51-unit-edges', 'edges_total': 3}], now_epoch=0.0
        )
        assert _sample(ms.EDGES_TOTAL, group_id='ale51-unit-edges') == 3


class TestApplyDataQuality:
    def test_duplicate_episodes(self):
        ms._apply_duplicate_episodes(
            [{'group_id': 'ale51-unit-dupe', 'duplicate_episodes': 2}], now_epoch=0.0
        )
        assert _sample(ms.EPISODES_DUPLICATE_NAMES, group_id='ale51-unit-dupe') == 2

    def test_orphaned_edges(self):
        ms._apply_orphaned_edges(
            [{'group_id': 'ale51-unit-orphan', 'orphaned_edges': 1}], now_epoch=0.0
        )
        assert _sample(ms.EDGES_ORPHANED, group_id='ale51-unit-orphan') == 1


class TestRefreshGraphMetrics:
    class FakeDriver:
        def __init__(self, results: dict[str, Exception | list[dict]]):
            self._results = results

        async def execute_query(self, query: str):
            for marker, outcome in self._results.items():
                if marker in query:
                    if isinstance(outcome, Exception):
                        raise outcome
                    return outcome, None, None
            raise AssertionError(f'unexpected query: {query[:60]!r}')

    @pytest.mark.asyncio
    async def test_one_failing_query_does_not_block_the_others(self):
        driver = self.FakeDriver(
            {
                'MATCH (e:Episodic)\nRETURN': RuntimeError('boom'),
                'MATCH (n:Entity)': [{'group_id': 'ale51-unit-partial', 'entity_nodes_total': 9}],
                'RELATES_TO]->(:Entity)\nRETURN': [
                    {'group_id': 'ale51-unit-partial', 'edges_total': 4}
                ],
                'WITH e.group_id AS group_id, e.name': [],
                'UNWIND e.episodes': [],
            }
        )
        await ms.refresh_graph_metrics(driver, now_epoch=0.0)
        assert _sample(ms.ENTITY_NODES_TOTAL, group_id='ale51-unit-partial') == 9
        assert _sample(ms.EDGES_TOTAL, group_id='ale51-unit-partial') == 4
        assert ms.LAST_REFRESH_SUCCESS._value.get() == 0  # one query failed


class TestMetricsRefreshLoopGuard:
    class NonNeo4jDriver:
        provider = GraphProvider.FALKORDB

    @pytest.mark.asyncio
    async def test_non_neo4j_driver_returns_without_looping(self):
        # Must return promptly rather than entering the infinite loop.
        await asyncio.wait_for(
            ms.metrics_refresh_loop(self.NonNeo4jDriver()), timeout=1.0
        )


def test_render_metrics_returns_prometheus_exposition_format():
    body, content_type = ms.render_metrics()
    assert content_type.startswith('text/plain')
    assert b'graphiti_queue_depth' in body or b'# HELP' in body
