# Storing Embeddings

The `WordPress\AI\Embeddings` namespace provides a portable storage layer for embedding vectors. It is persistence only: generating vectors is the job of the AI Client, and similarity search is built on top of it by higher-level code.

Vectors are stored in the `wpai_embeddings` table, one row per `(object_type, object_id, provider, model, chunk_index)`. Embedding vectors are only comparable to other vectors produced by the same model, so the provider and model are part of every record's identity — an index can never be queried with vectors from a different model by accident. The table is created on the first write, never by a read.

```php
use WordPress\AI\Embeddings\Embedding_Record;
use WordPress\AI\Embeddings\Embedding_Repository;

$repository = new Embedding_Repository();

// Store (or replace) the vector for post 42 produced by a specific model.
$repository->save(
	new Embedding_Record( 'post', 42, 'ollama', 'nomic-embed-text:latest', $vector, 0, hash( 'sha256', $text ) )
);

// Read it back — always scoped to the model that produced it.
$records = $repository->get( 'post', 42, 'ollama', 'nomic-embed-text:latest' );

// Cheap staleness check before re-embedding.
$stale = $repository->get_content_hash( 'post', 42, 'ollama', 'nomic-embed-text:latest' ) !== hash( 'sha256', $text );

// Bounded, newest-first lookup of indexed objects, and a batched scan over every vector for a model.
$post_ids = $repository->get_object_ids( 'post', 'ollama', 'nomic-embed-text:latest', 500 );
foreach ( $repository->iterate( 'ollama', 'nomic-embed-text:latest', 'post' ) as $record ) {
	// $record->get_vector(), $record->get_object_id() …
}

// Switching models means re-indexing; drop everything the old model produced.
$repository->delete_for_model( 'ollama', 'nomic-embed-text:latest' );
```

Longer content can be stored as several chunks of the same object by using `chunk_index`; `get()` returns them in chunk order. Vectors are packed as little-endian float32 bytes (see `Vector_Codec`), the same layout MariaDB's native `VECTOR` type uses, so a backend with a native vector index can implement `Embedding_Repository_Interface` against the same data.
