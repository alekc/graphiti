"""Runtime patch: wire use_combined_extraction through add_episode_bulk.

configs/llm.yml's Dockerfile.standalone build deliberately installs
graphiti-core from PyPI (stripping any local path override and regenerating
uv.lock against the pin), so a source-tree edit under graphiti_core/ in this
monorepo never reaches the built image; only mcp_server/'s own src/ does.
This module is the runtime side of that same fix, applied against whatever
graphiti-core version PyPI installed.

The published graphiti-core==0.29.3 wheel already ships the underlying
dispatch machinery: graphiti_core.utils.bulk_utils.extract_nodes_and_edges_bulk
takes use_combined_extraction and picks the combined single-LLM-call path or
the separate two-call path accordingly. What it does not ship is a way to
reach that flag: Graphiti.add_episode_bulk's own public signature has no
use_combined_extraction parameter, and its private helper
_extract_and_dedupe_nodes_bulk never forwards one either. Both are grafted
here, verbatim from the same functions with exactly one parameter added and
threaded through one call site each, following the same
compile-and-bind-onto-the-live-class technique already used for the litellm
chatgpt provider patches on this cluster (see litellm_chatgpt_patch.py in
terraform-home-cluster): safe here because neither function calls bare
super(), which is the one thing that technique cannot graft.

Self-deprecating like those patches: if a future graphiti-core release adds
use_combined_extraction to add_episode_bulk's own signature, apply() detects
it via inspect.signature and skips, so bumping GRAPHITI_CORE_VERSION forward
past that release makes this file inert rather than wrong.
"""

import inspect
import logging

logger = logging.getLogger(__name__)

# Verbatim from graphiti_core/graphiti.py on the alekc/graphiti fork
# (main @ 993e081, itself unmodified from the graphiti-core==0.29.3 wheel at
# this range except for the use_combined_extraction lines below), with
# `self` typed loosely on purpose: this is bound as a plain function
# attribute on the live class, not textually nested in one.
_EXTRACT_AND_DEDUPE_SRC = '''
async def _patched_extract_and_dedupe_nodes_bulk(
    self,
    episode_context: list[tuple[EpisodicNode, list[EpisodicNode]]],
    edge_type_map: dict[tuple[str, str], list[str]],
    edge_types: dict[str, type[BaseModel]] | None,
    entity_types: dict[str, type[BaseModel]] | None,
    excluded_entity_types: list[str] | None,
    custom_extraction_instructions: str | None = None,
    use_combined_extraction: bool = False,
) -> tuple[
    dict[str, list[EntityNode]],
    dict[str, str],
    list[list[EntityEdge]],
]:
    """Extract nodes and edges from all episodes and deduplicate."""
    # Extract all nodes and edges for each episode
    extracted_nodes_bulk, extracted_edges_bulk = await extract_nodes_and_edges_bulk(
        self.clients,
        episode_context,
        edge_type_map=edge_type_map,
        edge_types=edge_types,
        entity_types=entity_types,
        excluded_entity_types=excluded_entity_types,
        custom_extraction_instructions=custom_extraction_instructions,
        use_combined_extraction=use_combined_extraction,
    )

    # Dedupe extracted nodes in memory
    nodes_by_episode, uuid_map = await dedupe_nodes_bulk(
        self.clients, extracted_nodes_bulk, episode_context, entity_types
    )

    return nodes_by_episode, uuid_map, extracted_edges_bulk
'''

_ADD_EPISODE_BULK_SRC = '''
async def _patched_add_episode_bulk(
    self,
    bulk_episodes: list[RawEpisode],
    group_id: str | None = None,
    entity_types: dict[str, type[BaseModel]] | None = None,
    excluded_entity_types: list[str] | None = None,
    edge_types: dict[str, type[BaseModel]] | None = None,
    edge_type_map: dict[tuple[str, str], list[str]] | None = None,
    custom_extraction_instructions: str | None = None,
    saga: str | SagaNode | None = None,
    use_combined_extraction: bool = False,
) -> AddBulkEpisodeResults:
    """Process multiple episodes in bulk and update the graph.

    Patched copy of Graphiti.add_episode_bulk adding use_combined_extraction;
    see graphiti_core_combined_extraction.py's module docstring for why this
    is a runtime patch rather than a source-tree one.
    """
    with self.tracer.start_span('add_episode_bulk') as bulk_span:
        bulk_span.add_attributes({'episode.count': len(bulk_episodes)})

        try:
            start = time()
            now = utc_now()

            # if group_id is None, use the default group id by the provider
            if group_id is None:
                group_id = get_default_group_id(self.driver.provider)
            else:
                validate_group_id(group_id)
                if group_id != self.driver._database:
                    # if group_id is provided, use it as the database name
                    self.driver = self.driver.clone(database=group_id)
                    self.clients.driver = self.driver

            # Create default edge type map
            edge_type_map_default = (
                {('Entity', 'Entity'): list(edge_types.keys())}
                if edge_types is not None
                else {('Entity', 'Entity'): []}
            )

            episodes = [
                await EpisodicNode.get_by_uuid(self.driver, episode.uuid)
                if episode.uuid is not None
                else EpisodicNode(
                    name=episode.name,
                    labels=[],
                    source=episode.source,
                    content=episode.content,
                    source_description=episode.source_description,
                    group_id=group_id,
                    created_at=now,
                    valid_at=episode.reference_time,
                )
                for episode in bulk_episodes
            ]

            # Save all episodes
            await add_nodes_and_edges_bulk(
                driver=self.driver,
                episodic_nodes=episodes,
                episodic_edges=[],
                entity_nodes=[],
                entity_edges=[],
                embedder=self.embedder,
            )

            # Get previous episode context for each episode
            episode_context = await retrieve_previous_episodes_bulk(self.driver, episodes)

            # Extract and dedupe nodes and edges
            (
                nodes_by_episode,
                uuid_map,
                extracted_edges_bulk,
            ) = await self._extract_and_dedupe_nodes_bulk(
                episode_context,
                edge_type_map or edge_type_map_default,
                edge_types,
                entity_types,
                excluded_entity_types,
                custom_extraction_instructions,
                use_combined_extraction,
            )

            # Create Episodic Edges
            episodic_edges: list[EpisodicEdge] = []
            for episode_uuid, nodes in nodes_by_episode.items():
                episodic_edges.extend(build_episodic_edges(nodes, episode_uuid, now))

            # Re-map edge pointers and dedupe edges
            extracted_edges_bulk_updated: list[list[EntityEdge]] = [
                resolve_edge_pointers(edges, uuid_map) for edges in extracted_edges_bulk
            ]

            edges_by_episode = await dedupe_edges_bulk(
                self.clients,
                extracted_edges_bulk_updated,
                episode_context,
                [],
                edge_types or {},
                edge_type_map or edge_type_map_default,
            )

            # Resolve nodes and edges against the existing graph
            (
                final_hydrated_nodes,
                resolved_edges,
                invalidated_edges,
                final_uuid_map,
            ) = await self._resolve_nodes_and_edges_bulk(
                nodes_by_episode,
                edges_by_episode,
                episode_context,
                entity_types,
                edge_types,
                edge_type_map or edge_type_map_default,
                episodes,
            )

            # Resolved pointers for episodic edges
            resolved_episodic_edges = resolve_edge_pointers(episodic_edges, final_uuid_map)

            # save data to KG
            await add_nodes_and_edges_bulk(
                self.driver,
                episodes,
                resolved_episodic_edges,
                final_hydrated_nodes,
                resolved_edges + invalidated_edges,
                self.embedder,
            )

            # Handle saga association if provided
            if saga is not None:
                # Get or create saga node based on input type
                if isinstance(saga, str):
                    # Anchor a newly minted saga to the earliest episode
                    # reference time in the bulk, so created_at reflects the
                    # episode window rather than the time this run started.
                    valid_ats = [ep.valid_at for ep in episodes if ep.valid_at is not None]
                    saga_created_at = min(valid_ats) if valid_ats else now
                    saga_node = await self._get_or_create_saga(saga, group_id, saga_created_at)
                else:
                    saga_node = saga

                # Sort episodes by valid_at to create NEXT_EPISODE chain in correct order
                sorted_episodes = sorted(episodes, key=lambda e: e.valid_at)

                # Find the most recent episode already in the saga
                previous_episode_uuid = await self._saga_get_previous_episode_uuid(
                    saga_node.uuid, ''
                )

                for episode in sorted_episodes:
                    # Create NEXT_EPISODE edge from the previous episode
                    if previous_episode_uuid is not None:
                        next_episode_edge = NextEpisodeEdge(
                            source_node_uuid=previous_episode_uuid,
                            target_node_uuid=episode.uuid,
                            group_id=group_id,
                            created_at=now,
                        )
                        await next_episode_edge.save(self.driver)

                    # Create HAS_EPISODE edge from saga to episode
                    has_episode_edge = HasEpisodeEdge(
                        source_node_uuid=saga_node.uuid,
                        target_node_uuid=episode.uuid,
                        group_id=group_id,
                        created_at=now,
                    )
                    await has_episode_edge.save(self.driver)

                    # Update previous_episode_uuid for the next iteration
                    previous_episode_uuid = episode.uuid

                # Track first and last episode on the saga node
                if sorted_episodes:
                    if saga_node.first_episode_uuid is None:
                        saga_node.first_episode_uuid = sorted_episodes[0].uuid
                    saga_node.last_episode_uuid = sorted_episodes[-1].uuid
                    await saga_node.save(self.driver)

            end = time()

            # Add span attributes
            bulk_span.add_attributes(
                {
                    'group_id': group_id,
                    'node.count': len(final_hydrated_nodes),
                    'edge.count': len(resolved_edges + invalidated_edges),
                    'duration_ms': (end - start) * 1000,
                }
            )

            logger.info(f'Completed add_episode_bulk in {(end - start) * 1000} ms')

            return AddBulkEpisodeResults(
                episodes=episodes,
                episodic_edges=resolved_episodic_edges,
                nodes=final_hydrated_nodes,
                edges=resolved_edges + invalidated_edges,
                communities=[],
                community_edges=[],
            )

        except Exception as e:
            bulk_span.set_status('error', str(e))
            bulk_span.record_exception(e)
            raise e
'''


def apply() -> None:
    """Graft use_combined_extraction onto Graphiti.add_episode_bulk, once.

    No-ops (logging why) if upstream has already shipped this, or if the
    installed graphiti_core does not match the shape this patch assumes.
    """
    from graphiti_core.graphiti import Graphiti

    if 'use_combined_extraction' in inspect.signature(Graphiti.add_episode_bulk).parameters:
        logger.info(
            'graphiti_core_combined_extraction: add_episode_bulk already supports '
            'use_combined_extraction upstream, patch not needed'
        )
        return

    import graphiti_core.graphiti as graphiti_module

    try:
        namespace = vars(graphiti_module)
        exec(compile(_EXTRACT_AND_DEDUPE_SRC, __name__ + ':_extract_and_dedupe', 'exec'), namespace)
        exec(compile(_ADD_EPISODE_BULK_SRC, __name__ + ':add_episode_bulk', 'exec'), namespace)

        Graphiti._extract_and_dedupe_nodes_bulk = namespace[
            '_patched_extract_and_dedupe_nodes_bulk'
        ]
        Graphiti.add_episode_bulk = namespace['_patched_add_episode_bulk']
    except Exception:
        logger.exception(
            'graphiti_core_combined_extraction: failed to graft use_combined_extraction, '
            'add_episode_bulk stays unpatched (use_combined_extraction requests will be '
            'silently ignored by the installed graphiti_core, not error)'
        )
        return

    logger.info(
        'graphiti_core_combined_extraction: patched Graphiti.add_episode_bulk and '
        '_extract_and_dedupe_nodes_bulk to accept use_combined_extraction'
    )
