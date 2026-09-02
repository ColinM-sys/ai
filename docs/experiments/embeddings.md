# Storing Embeddings

The `WordPress\AI\Embeddings` namespace provides a portable storage layer for embedding vectors. It is persistence only: generating vectors is the job of the AI Client, and similarity search is built on top of it by higher-level code.

Vectors are stored in the `wpai_embeddings` table, one row per `(object_type, object_id, provider, model, chunk_index)`. Embedding vectors are only comparable to other vectors produced by the same model, so the provider and model are part of every record's identity — an index can never be queried with vectors from a different model by accident. The table is created on the first write, never by a read.

```php
use WordPress\AI\Embeddings\Embedding_Record;
use WordPress\AI\Embeddings\Embedding_Repository;
use function WordPress\AI\generate_embeddings;

$repository = new Embedding_Repository();

$result = generate_embeddings( $text, array(
	'provider' => 'ollama',
	'model'    => 'nomic-embed-text:latest',
) );

if ( is_wp_error( $result ) ) {
	return $result;
}

// Take the provider and model from the result rather than from the request. They are part of the
// row's identity, so they have to name what actually produced the vector — and when a model
// instance is passed instead of an ID, the caller never named a provider in the first place.
$repository->save(
	new Embedding_Record(
		'post',
		42,
		$result->getProviderMetadata()->getId(),
		$result->getModelMetadata()->getId(),
		$result->getEmbeddings()[0]->getValues(),
		0,
		hash( 'sha256', $text )
	)
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

// Switching models means re-indexing. Provider and model are part of every row's identity, so
// index the new model alongside the old one and only drop the old vectors once coverage is
// complete — the existing index keeps serving results for the whole backfill.
foreach ( $post_ids as $post_id ) {
	$repository->save(
		new Embedding_Record( 'post', $post_id, 'ollama', 'mxbai-embed-large:latest', $new_vector )
	);
}

// Cut over only after the new model covers everything.
$repository->delete_for_model( 'ollama', 'nomic-embed-text:latest' );
```

Longer content can be stored as several chunks of the same object by using `chunk_index`; `get()` returns them in chunk order. Vectors are packed as little-endian float32 bytes (see `Vector_Codec`), the same layout MariaDB's native `VECTOR` type uses, so a backend with a native vector index can implement `Embedding_Repository_Interface` against the same data.
