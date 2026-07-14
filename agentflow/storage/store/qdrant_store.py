"""
Qdrant Vector Store Implementation for TAF Framework

This module provides a modern, async-first implementation of BaseStore using Qdrant
as the backend vector database. Supports both local and cloud Qdrant deployments
with configurable embedding services.
"""

import asyncio
import logging
from datetime import datetime
from typing import Any

from agentflow.core.state import Message

from .base_store import BaseStore
from .embedding.base_embedding import BaseEmbedding
from .store_schema import (
    DistanceMetric,
    MemoryRecord,
    MemorySearchResult,
    MemoryType,
    RetrievalStrategy,
)


HAS_QDRANT = False

try:
    import qdrant_client  # noqa: F401

    HAS_QDRANT = True
except ImportError:
    pass

logger = logging.getLogger("agentflow.store")

# Default collection name used when none is specified via config or constructor.
# Centralised here so factory functions and the class share the same value
# without duplicating the magic string.
DEFAULT_COLLECTION = "agentflow_memories"


class QdrantStore(BaseStore):
    """
    Modern async-first Qdrant-based vector store implementation.

    Features:
    - Async-only operations for better performance
    - Local and cloud Qdrant deployment support
    - Configurable embedding services
    - Efficient vector similarity search with multiple strategies
    - Collection management with automatic creation
    - Rich metadata filtering capabilities
    - User and agent-scoped operations

    Example:
        ```python
        # Local Qdrant with OpenAI embeddings
        store = QdrantStore(path="./qdrant_data", embedding_service=OpenAIEmbeddingService())

        # Remote Qdrant
        store = QdrantStore(host="localhost", port=6333, embedding_service=OpenAIEmbeddingService())

        # Cloud Qdrant
        store = QdrantStore(
            url="https://xyz.qdrant.io",
            api_key="your-api-key",
            embedding_service=OpenAIEmbeddingService(),
        )
        ```
    """

    def __init__(
        self,
        embedding: BaseEmbedding,
        path: str | None = None,
        host: str | None = None,
        port: int | None = None,
        url: str | None = None,
        api_key: str | None = None,
        collection: str | None = None,
        distance_metric: DistanceMetric = DistanceMetric.COSINE,
        **kwargs: Any,
    ):
        """
        Initialize Qdrant vector store.

        Args:
            embedding: Service for generating embeddings
            path: Path for local Qdrant (file-based storage)
            host: Host for remote Qdrant server
            port: Port for remote Qdrant server
            url: URL for Qdrant cloud
            api_key: API key for Qdrant cloud
            collection: Collection name to use (defaults to DEFAULT_COLLECTION)
            distance_metric: Default distance metric
            **kwargs: Additional client parameters
        """
        if not HAS_QDRANT:
            raise ImportError(
                "qdrant-client package is required for QdrantStore. "
                "Install with `pip install 'agentflow[qdrant]'`."
            )
        self.embedding = embedding

        # Initialize async client
        from qdrant_client import AsyncQdrantClient

        if path:
            self.client = AsyncQdrantClient(path=path, **kwargs)
        elif url:
            self.client = AsyncQdrantClient(url=url, api_key=api_key, **kwargs)
        else:
            host = host or "localhost"
            port = port or 6333
            self.client = AsyncQdrantClient(host=host, port=port, api_key=api_key, **kwargs)

        # Cache for collection existence checks, so the network round-trip happens
        # once per collection rather than on every store/search/get.
        self._collection_cache = set()
        # Serializes the cold-start miss: without it, N concurrent first-requests
        # all miss the cache together and each issues its own create_collection.
        self._collection_lock = asyncio.Lock()
        self._setup_lock = asyncio.Lock()

        self.collection = collection or DEFAULT_COLLECTION
        self._default_distance_metric = distance_metric

        logger.info(f"Initialized QdrantStore with config: path={path}, host={host}, url={url}")

    async def asetup(self) -> Any:
        """Set up the store and ensure default collection exists."""
        async with self._setup_lock:
            await self._ensure_collection_exists(self.collection)
        return True

    def _distance_metric_to_qdrant(self, metric: DistanceMetric):
        """Convert framework distance metric to Qdrant distance."""
        from qdrant_client.http.models import Distance

        mapping = {
            DistanceMetric.COSINE: Distance.COSINE,
            DistanceMetric.EUCLIDEAN: Distance.EUCLID,
            DistanceMetric.DOT_PRODUCT: Distance.DOT,
            DistanceMetric.MANHATTAN: Distance.MANHATTAN,
        }
        return mapping.get(metric, Distance.COSINE)

    def _extract_config_values(self, config: dict[str, Any]) -> tuple[str | None, str | None, str]:
        """Extract user_id, thread_id, and collection from config."""
        user_id = config.get("user_id")
        thread_id = config.get("thread_id")
        collection = config.get("collection", self.collection)
        return user_id, thread_id, collection

    def _point_to_search_result(self, point) -> MemorySearchResult:
        """Convert Qdrant point to MemorySearchResult."""
        payload = getattr(point, "payload", {}) or {}

        # Extract content
        content = payload.get("content", "")

        # Convert memory_type string back to enum
        memory_type_str = payload.get("memory_type", "episodic")
        try:
            memory_type = MemoryType(memory_type_str)
        except ValueError:
            memory_type = MemoryType.EPISODIC

        # Parse timestamp
        timestamp_str = payload.get("timestamp")
        timestamp = None
        if timestamp_str:
            try:
                timestamp = datetime.fromisoformat(timestamp_str)
            except (ValueError, TypeError):
                timestamp = None

        return MemorySearchResult(
            id=str(point.id),
            content=content,
            score=float(getattr(point, "score", 1.0) or 0.0),
            memory_type=memory_type,
            metadata=payload,
            vector=getattr(point, "vector", None),
            user_id=payload.get("user_id"),
            thread_id=payload.get("thread_id")
            or payload.get("agent_id"),  # Support both thread_id and agent_id
            timestamp=timestamp,
        )

    def _build_qdrant_filter(
        self,
        user_id: str | None = None,
        thread_id: str | None = None,
        memory_type: MemoryType | None = None,
        category: str | None = None,
        filters: dict[str, Any] | None = None,
    ) -> Any | None:
        """Build Qdrant filter from parameters."""
        conditions = []

        from qdrant_client.http.models import (
            FieldCondition,
            Filter,
            MatchValue,
        )

        # Add user/agent filters
        if user_id:
            conditions.append(
                FieldCondition(
                    key="user_id",
                    match=MatchValue(value=user_id),
                ),
            )
        if thread_id:
            conditions.append(
                FieldCondition(
                    key="thread_id",
                    match=MatchValue(value=thread_id),
                ),
            )
        if memory_type:
            conditions.append(
                FieldCondition(
                    key="memory_type",
                    match=MatchValue(value=memory_type.value),
                )
            )
        if category:
            conditions.append(FieldCondition(key="category", match=MatchValue(value=category)))

        # Add custom filters
        if filters:
            for key, value in filters.items():
                if isinstance(value, str | int | bool):
                    conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))

        return Filter(must=conditions) if conditions else None

    async def _ensure_collection_exists(self, collection: str) -> None:
        """Ensure collection exists, create if not.

        The `_collection_cache` fast path means the network round-trip happens once
        per collection, not on every store/search/get. The lock closes the cold-start
        race: without it, N concurrent first-requests all missed the cache together
        and each issued its own get_collections()/create_collection() call.
        """
        if collection in self._collection_cache:
            return

        from qdrant_client.http.models import PayloadSchemaType, VectorParams

        async with self._collection_lock:
            # Re-check under the lock: another coroutine may have ensured it while
            # we were waiting.
            if collection in self._collection_cache:
                return

            await self._ensure_collection_exists_locked(
                collection, PayloadSchemaType, VectorParams
            )

    async def _ensure_collection_exists_locked(
        self,
        collection: str,
        PayloadSchemaType,  # noqa: N803
        VectorParams,  # noqa: N803
    ) -> None:
        """Create the collection and its payload indexes. Caller holds the lock."""
        try:
            # Check if collection exists
            collections = await self.client.get_collections()
            existing_names = {col.name for col in collections.collections}

            if collection not in existing_names:
                # Create collection with vector configuration
                await self.client.create_collection(
                    collection_name=collection,
                    vectors_config=VectorParams(
                        size=self.embedding.dimension,
                        distance=self._distance_metric_to_qdrant(
                            self._default_distance_metric,
                        ),
                    ),
                )
                logger.info(f"Created collection: {collection}")

            # Ensure payload indexes exist for filterable fields.
            # Required for remote/cloud Qdrant where full scan is disabled.
            # Qdrant's create_payload_index is idempotent — calling it on an
            # already-indexed field is a no-op, so this is safe for both new
            # and pre-existing collections.
            for field_name in (
                "user_id",
                "memory_type",
                "category",
                "memory_key",
            ):
                try:
                    await self.client.create_payload_index(
                        collection_name=collection,
                        field_name=field_name,
                        field_schema=PayloadSchemaType.KEYWORD,
                    )
                except Exception:
                    # Best-effort: log and continue so one failing index
                    # doesn't prevent the collection from being usable.
                    logger.debug(
                        "Payload index for '%s' on '%s' may already exist or failed",
                        field_name,
                        collection,
                    )
            logger.info(f"Ensured payload indexes for collection: {collection}")

            self._collection_cache.add(collection)
        except Exception as e:
            logger.error(f"Error ensuring collection {collection} exists: {e}")
            raise

    def _prepare_content(self, content: str | Message) -> str:
        """Extract text content from string or Message."""
        if isinstance(content, Message):
            return content.text()
        return content

    def _create_memory_record(
        self,
        content: str | Message,
        user_id: str | None = None,
        thread_id: str | None = None,
        memory_type: MemoryType = MemoryType.EPISODIC,
        category: str = "general",
        metadata: dict[str, Any] | None = None,
    ) -> MemoryRecord:
        """Create a memory record from parameters."""
        text_content = self._prepare_content(content)

        if isinstance(content, Message):
            return MemoryRecord.from_message(
                content,
                user_id=user_id,
                thread_id=thread_id,
                additional_metadata=metadata,
            )

        return MemoryRecord(
            content=text_content,
            user_id=user_id,
            thread_id=thread_id,
            memory_type=memory_type,
            metadata=metadata or {},
            category=category,
        )

    # --- BaseStore abstract method implementations ---

    async def astore(
        self,
        config: dict[str, Any],
        content: str | Message,
        memory_type: MemoryType = MemoryType.EPISODIC,
        category: str = "general",
        metadata: dict[str, Any] | None = None,
        **kwargs,
    ) -> str:
        """Store a new memory."""
        user_id, thread_id, collection = self._extract_config_values(config)

        from qdrant_client.http.models import PointStruct

        # Ensure collection exists
        await self._ensure_collection_exists(collection)

        record = self._create_memory_record(
            content=content,
            user_id=user_id,
            thread_id=thread_id,
            memory_type=memory_type,
            category=category,
            metadata=metadata,
        )
        record.id = kwargs.get("memory_id") or await self.generate_framework_id()

        # Generate embedding
        text_content = self._prepare_content(content)
        vector = await self.embedding.aembed(text_content)
        if not vector or len(vector) != self.embedding.dimension:
            raise ValueError("Embedding service returned invalid vector")

        # Prepare payload
        payload = {
            "content": record.content,
            "user_id": record.user_id,
            "thread_id": record.thread_id,
            "memory_type": record.memory_type.value,
            "category": record.category,
            "timestamp": record.timestamp.isoformat() if record.timestamp else None,
            **record.metadata,
        }

        # Create point
        point = PointStruct(
            id=record.id,
            vector=vector,
            payload=payload,
        )

        # Store in Qdrant
        await self.client.upsert(
            collection_name=collection,
            points=[point],
        )

        logger.debug(f"Stored memory {record.id} in collection {collection}")
        return record.id

    async def asearch(
        self,
        config: dict[str, Any],
        query: str,
        memory_type: MemoryType | None = None,
        category: str | None = None,
        limit: int = 10,
        score_threshold: float | None = None,
        filters: dict[str, Any] | None = None,
        retrieval_strategy: RetrievalStrategy = RetrievalStrategy.SIMILARITY,
        distance_metric: DistanceMetric = DistanceMetric.COSINE,
        max_tokens: int = 4000,
        **kwargs,
    ) -> list[MemorySearchResult]:
        """Search memories by content similarity.

        Long-term memory is cross-thread — thread_id is intentionally excluded
        from the search filter so results span all conversations for the user.
        """
        user_id, _thread_id, collection = self._extract_config_values(config)

        # Ensure collection exists
        await self._ensure_collection_exists(collection)

        # Generate query embedding
        query_vector = await self.embedding.aembed(query)
        if not query_vector or len(query_vector) != self.embedding.dimension:
            raise ValueError("Embedding service returned invalid vector")

        # Build filter — search by user only; thread_id is excluded
        search_filter = self._build_qdrant_filter(
            user_id=user_id,
            memory_type=memory_type,
            category=category,
            filters=filters,
        )

        # Perform search
        search_result = await self.client.search(
            collection_name=collection,
            query_vector=query_vector,
            query_filter=search_filter,
            limit=limit,
            score_threshold=score_threshold,
        )

        # Convert to search results
        results = [self._point_to_search_result(point) for point in search_result]

        logger.debug(f"Found {len(results)} memories for query in collection {collection}")
        return results

    async def aget(
        self,
        config: dict[str, Any],
        memory_id: str,
        **kwargs,
    ) -> MemorySearchResult | None:
        """Get a specific memory by ID."""
        user_id, _thread_id, collection = self._extract_config_values(config)

        try:
            # Ensure collection exists
            await self._ensure_collection_exists(collection)

            # Get point by ID
            points = await self.client.retrieve(
                collection_name=collection,
                ids=[memory_id],
            )

            if not points:
                return None

            point = points[0]
            result = self._point_to_search_result(point)

            # Verify user access only — thread_id is intentionally not checked
            # because long-term memory is cross-thread.
            if user_id and result.user_id != user_id:
                return None

            return result

        except Exception as e:
            logger.error(f"Error retrieving memory {memory_id}: {e}")
            return None

    async def aget_all(
        self,
        config: dict[str, Any],
        limit: int = 100,
        **kwargs,
    ) -> list[MemorySearchResult]:
        """Get all memories for a user."""
        user_id, _, collection = self._extract_config_values(config)

        # Ensure collection exists
        await self._ensure_collection_exists(collection)

        # Build filter
        search_filter = self._build_qdrant_filter(
            user_id=user_id,
        )

        # Perform search
        search_result = await self.client.search(
            collection_name=collection,
            query_vector=[],
            query_filter=search_filter,
            limit=limit,
        )

        # Convert to search results
        results = [self._point_to_search_result(point) for point in search_result]

        logger.debug(f"Found {len(results)} memories for query in collection {collection}")
        return results

    async def aupdate(
        self,
        config: dict[str, Any],
        memory_id: str,
        content: str | Message,
        metadata: dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        """Update an existing memory."""
        from qdrant_client.http.models import PointStruct

        user_id, _thread_id, collection = self._extract_config_values(config)

        # Get existing memory
        existing = await self.aget(config, memory_id)
        if not existing:
            raise ValueError(f"Memory {memory_id} not found")

        # Verify user access only — thread_id is not an access-control boundary
        # for long-term memory (memories are shared across all threads for a user).
        if user_id and existing.user_id != user_id:
            raise PermissionError("User does not have permission to update this memory")

        # Prepare new content
        text_content = self._prepare_content(content)
        new_vector = await self.embedding.aembed(text_content)
        if not new_vector or len(new_vector) != self.embedding.dimension:
            raise ValueError("Embedding service returned invalid vector")

        # Update payload
        replace_metadata = kwargs.get("replace_metadata", False)
        if replace_metadata:
            updated_metadata = metadata or {}
        else:
            updated_metadata = {**existing.metadata}
            if metadata:
                updated_metadata.update(metadata)

        # Strip system fields from metadata so they don't overwrite the
        # explicit values we set below (existing.metadata mirrors the full
        # Qdrant payload, including "content").
        _system_keys = {
            "content",
            "user_id",
            "thread_id",
            "memory_type",
            "category",
            "timestamp",
        }
        extra_metadata = {k: v for k, v in updated_metadata.items() if k not in _system_keys}

        updated_payload = {
            **extra_metadata,
            "content": text_content,
            "user_id": existing.user_id,
            "thread_id": existing.thread_id,
            "memory_type": existing.memory_type.value,
            "category": updated_metadata.get("category", "general"),
            "timestamp": datetime.now().isoformat(),
        }

        # Create updated point
        point = PointStruct(
            id=memory_id,
            vector=new_vector,
            payload=updated_payload,
        )

        # Update in Qdrant
        await self.client.upsert(
            collection_name=collection,
            points=[point],
        )

        logger.debug(f"Updated memory {memory_id} in collection {collection}")

    async def adelete(
        self,
        config: dict[str, Any],
        memory_id: str,
        **kwargs,
    ) -> None:
        """Delete a memory by ID."""
        from qdrant_client.http import models

        user_id, _thread_id, collection = self._extract_config_values(config)

        # Verify memory exists and user has access
        existing = await self.aget(config, memory_id)
        if not existing:
            raise ValueError(f"Memory {memory_id} not found")

        # Verify user access only — thread_id is not an access-control boundary
        # for long-term memory.
        if user_id and existing.user_id != user_id:
            raise PermissionError("User does not have permission to delete this memory")

        # Delete from Qdrant
        await self.client.delete(
            collection_name=collection,
            points_selector=models.PointIdsList(points=[memory_id]),
        )

        logger.debug(f"Deleted memory {memory_id} from collection {collection}")

    async def aforget_memory(
        self,
        config: dict[str, Any],
        **kwargs,
    ) -> None:
        """Delete all memories for a user or agent."""
        from qdrant_client.http import models

        user_id, agent_id, collection = self._extract_config_values(config)

        # Build filter for memories to delete
        delete_filter = self._build_qdrant_filter(user_id=user_id, thread_id=agent_id)

        if delete_filter:
            # Delete matching memories
            await self.client.delete(
                collection_name=collection,
                points_selector=models.FilterSelector(filter=delete_filter),
            )

            logger.info(
                f"Deleted all memories for user_id={user_id}, agent_id={agent_id} "
                f"in collection {collection}"
            )
        else:
            logger.warning("No user_id or agent_id specified for memory deletion")

    async def arelease(self) -> None:
        """Clean up resources."""
        if hasattr(self.client, "close"):
            await self.client.close()
        logger.info("QdrantStore resources released")


# Convenience factory functions


def create_local_qdrant_store(
    path: str,
    embedding: BaseEmbedding,
    collection: str = DEFAULT_COLLECTION,
    **kwargs,
) -> QdrantStore:
    """Create a local Qdrant store."""
    return QdrantStore(
        embedding=embedding,
        path=path,
        collection=collection,
        **kwargs,
    )


def create_remote_qdrant_store(
    host: str,
    port: int,
    embedding: BaseEmbedding,
    collection: str = DEFAULT_COLLECTION,
    **kwargs,
) -> QdrantStore:
    """Create a remote Qdrant store."""
    return QdrantStore(
        embedding=embedding,
        host=host,
        port=port,
        collection=collection,
        **kwargs,
    )


def create_cloud_qdrant_store(
    url: str,
    api_key: str,
    embedding: BaseEmbedding,
    collection: str = DEFAULT_COLLECTION,
    **kwargs,
) -> QdrantStore:
    """Create a cloud Qdrant store."""
    return QdrantStore(
        embedding=embedding,
        url=url,
        api_key=api_key,
        collection=collection,
        **kwargs,
    )
