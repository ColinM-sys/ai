<?php
/**
 * Tests for the database-backed embedding repository.
 *
 * @package WordPress\AI\Tests\Integration\Includes\Embeddings
 */

namespace WordPress\AI\Tests\Integration\Includes\Embeddings;

use WP_UnitTestCase;
use WordPress\AI\Embeddings\Embedding_Record;
use WordPress\AI\Embeddings\Embedding_Repository;
use WordPress\AI\Embeddings\Embedding_Schema;

/**
 * Embedding_Repository test case.
 *
 * @since 1.4.0
 *
 * @covers \WordPress\AI\Embeddings\Embedding_Repository
 */
class Embedding_RepositoryTest extends WP_UnitTestCase {

	private const PROVIDER = 'ollama';
	private const MODEL    = 'nomic-embed-text:latest';

	/**
	 * Repository under test.
	 *
	 * @var \WordPress\AI\Embeddings\Embedding_Repository
	 */
	private Embedding_Repository $repository;

	/**
	 * Schema instance.
	 *
	 * @var \WordPress\AI\Embeddings\Embedding_Schema
	 */
	private Embedding_Schema $schema;

	/**
	 * Set up test case.
	 *
	 * @since 1.4.0
	 */
	protected function setUp(): void {
		parent::setUp();

		$this->schema = new Embedding_Schema();
		$this->schema->drop_table();

		$this->repository = new Embedding_Repository( $this->schema );
	}

	/**
	 * Tear down test case.
	 *
	 * @since 1.4.0
	 */
	protected function tearDown(): void {
		$this->schema->drop_table();

		parent::tearDown();
	}

	/**
	 * Builds a record with sensible defaults.
	 *
	 * @since 1.4.0
	 *
	 * @param int             $object_id   Object ID.
	 * @param list<int|float> $vector      Optional. Vector. Default a 3-component vector.
	 * @param string          $model       Optional. Model ID. Default the test model.
	 * @param int             $chunk_index Optional. Chunk index. Default 0.
	 * @param string          $hash        Optional. Content hash. Default empty.
	 * @param string          $object_type Optional. Object type. Default `post`.
	 * @param string          $provider    Optional. Provider ID. Default the test provider.
	 * @return \WordPress\AI\Embeddings\Embedding_Record The record.
	 */
	private function make_record(
		int $object_id,
		array $vector = array( 0.1, 0.2, 0.3 ),
		string $model = self::MODEL,
		int $chunk_index = 0,
		string $hash = '',
		string $object_type = 'post',
		string $provider = self::PROVIDER
	): Embedding_Record {
		return new Embedding_Record( $object_type, $object_id, $provider, $model, $vector, $chunk_index, $hash );
	}

	/**
	 * Tests that reads on a site that never stored an embedding do not create the table.
	 *
	 * @since 1.4.0
	 */
	public function test_reads_do_not_create_table(): void {
		$this->assertSame( array(), $this->repository->get( 'post', 1, self::PROVIDER, self::MODEL ) );
		$this->assertNull( $this->repository->get_by_id( 1 ) );
		$this->assertNull( $this->repository->get_content_hash( 'post', 1, self::PROVIDER, self::MODEL ) );
		$this->assertSame( array(), $this->repository->get_object_ids( 'post', self::PROVIDER, self::MODEL, 10 ) );
		$this->assertSame( 0, $this->repository->count_objects( 'post', self::PROVIDER, self::MODEL ) );
		$this->assertSame( array(), iterator_to_array( $this->repository->iterate( self::PROVIDER, self::MODEL ), false ) );
		$this->assertSame( 0, $this->repository->delete_for_object( 'post', 1 ) );
		$this->assertSame( 0, $this->repository->delete_for_model( self::PROVIDER, self::MODEL ) );

		$this->assertFalse( $this->schema->table_exists() );
	}

	/**
	 * Tests that the first write creates the table.
	 *
	 * @since 1.4.0
	 */
	public function test_save_creates_table_on_first_write(): void {
		$this->assertFalse( $this->schema->table_exists() );

		$this->repository->save( $this->make_record( 1 ) );

		$this->assertTrue( $this->schema->table_exists() );
	}

	/**
	 * Tests a save and read round trip.
	 *
	 * @since 1.4.0
	 */
	public function test_save_and_get_round_trip(): void {
		$vector = array( 0.123456, -0.654321, 1.0, 0.0 );
		$saved  = $this->repository->save( $this->make_record( 5, $vector, self::MODEL, 0, 'hash-5' ) );

		$this->assertGreaterThan( 0, $saved->get_id() );

		$records = $this->repository->get( 'post', 5, self::PROVIDER, self::MODEL );

		$this->assertCount( 1, $records );
		$record = $records[0];

		$this->assertSame( $saved->get_id(), $record->get_id() );
		$this->assertSame( 'post', $record->get_object_type() );
		$this->assertSame( 5, $record->get_object_id() );
		$this->assertSame( self::PROVIDER, $record->get_provider() );
		$this->assertSame( self::MODEL, $record->get_model() );
		$this->assertSame( 4, $record->get_dimensions() );
		$this->assertSame( 'hash-5', $record->get_content_hash() );
		$this->assertEqualsWithDelta( $vector, $record->get_vector(), 1.0e-6 );

		$by_id = $this->repository->get_by_id( $saved->get_id() );
		$this->assertNotNull( $by_id );
		$this->assertSame( 5, $by_id->get_object_id() );
	}

	/**
	 * Tests that saving again for the same object, model and chunk replaces the row in place.
	 *
	 * @since 1.4.0
	 */
	public function test_save_replaces_existing_vector_in_place(): void {
		global $wpdb;

		$first  = $this->repository->save( $this->make_record( 3, array( 0.1, 0.1 ), self::MODEL, 0, 'old' ) );
		$second = $this->repository->save( $this->make_record( 3, array( 0.9, 0.9 ), self::MODEL, 0, 'new' ) );

		$this->assertSame( $first->get_id(), $second->get_id() );

		$records = $this->repository->get( 'post', 3, self::PROVIDER, self::MODEL );
		$this->assertCount( 1, $records );
		$this->assertEqualsWithDelta( array( 0.9, 0.9 ), $records[0]->get_vector(), 1.0e-6 );
		$this->assertSame( 'new', $records[0]->get_content_hash() );

		$table = $this->schema->get_table_name();
		$this->assertSame( '1', $wpdb->get_var( "SELECT COUNT(*) FROM {$table}" ) ); // phpcs:ignore WordPress.DB.DirectDatabaseQuery.DirectQuery,WordPress.DB.DirectDatabaseQuery.NoCaching,WordPress.DB.PreparedSQL.InterpolatedNotPrepared
	}

	/**
	 * Tests that vectors from different models for the same object are kept apart.
	 *
	 * @since 1.4.0
	 */
	public function test_models_are_isolated(): void {
		$this->repository->save( $this->make_record( 1, array( 0.1, 0.2 ), 'nomic-embed-text:latest' ) );
		$this->repository->save( $this->make_record( 1, array( 0.5, 0.6, 0.7 ), 'mxbai-embed-large:latest' ) );
		$this->repository->save( $this->make_record( 1, array( 0.9 ), 'gemini-embedding-001', 0, '', 'post', 'google' ) );

		$nomic  = $this->repository->get( 'post', 1, self::PROVIDER, 'nomic-embed-text:latest' );
		$mxbai  = $this->repository->get( 'post', 1, self::PROVIDER, 'mxbai-embed-large:latest' );
		$gemini = $this->repository->get( 'post', 1, 'google', 'gemini-embedding-001' );

		$this->assertCount( 1, $nomic );
		$this->assertSame( 2, $nomic[0]->get_dimensions() );
		$this->assertCount( 1, $mxbai );
		$this->assertSame( 3, $mxbai[0]->get_dimensions() );
		$this->assertCount( 1, $gemini );
		$this->assertSame( 1, $gemini[0]->get_dimensions() );

		// Same model ID under a different provider is a different model.
		$this->assertSame( array(), $this->repository->get( 'post', 1, 'google', 'nomic-embed-text:latest' ) );

		$this->assertSame( 1, $this->repository->count_objects( 'post', self::PROVIDER, 'nomic-embed-text:latest' ) );
		$this->assertSame( 0, $this->repository->count_objects( 'post', self::PROVIDER, 'gemini-embedding-001' ) );
	}

	/**
	 * Tests that chunks are stored separately and returned in order.
	 *
	 * @since 1.4.0
	 */
	public function test_chunks_are_returned_in_order(): void {
		$this->repository->save_many(
			array(
				$this->make_record( 8, array( 0.3 ), self::MODEL, 2 ),
				$this->make_record( 8, array( 0.1 ), self::MODEL, 0 ),
				$this->make_record( 8, array( 0.2 ), self::MODEL, 1 ),
			)
		);

		$records = $this->repository->get( 'post', 8, self::PROVIDER, self::MODEL );

		$this->assertSame( array( 0, 1, 2 ), array_map( static fn( Embedding_Record $r ): int => $r->get_chunk_index(), $records ) );
		$this->assertSame( 1, $this->repository->count_objects( 'post', self::PROVIDER, self::MODEL ) );
	}

	/**
	 * Tests that object types are kept apart.
	 *
	 * @since 1.4.0
	 */
	public function test_object_types_are_isolated(): void {
		$this->repository->save( $this->make_record( 1, array( 0.1 ), self::MODEL, 0, '', 'post' ) );
		$this->repository->save( $this->make_record( 1, array( 0.2 ), self::MODEL, 0, '', 'term' ) );

		$this->assertCount( 1, $this->repository->get( 'post', 1, self::PROVIDER, self::MODEL ) );
		$this->assertCount( 1, $this->repository->get( 'term', 1, self::PROVIDER, self::MODEL ) );
		$this->assertSame( array( 1 ), $this->repository->get_object_ids( 'term', self::PROVIDER, self::MODEL, 10 ) );
	}

	/**
	 * Tests the content hash lookup.
	 *
	 * @since 1.4.0
	 */
	public function test_get_content_hash(): void {
		$this->repository->save( $this->make_record( 4, array( 0.1 ), self::MODEL, 0, 'sha-4' ) );

		$this->assertSame( 'sha-4', $this->repository->get_content_hash( 'post', 4, self::PROVIDER, self::MODEL ) );
		$this->assertNull( $this->repository->get_content_hash( 'post', 99, self::PROVIDER, self::MODEL ) );
		$this->assertNull( $this->repository->get_content_hash( 'post', 4, 'google', self::MODEL ) );
	}

	/**
	 * Tests the bounded, newest-first object ID lookup.
	 *
	 * @since 1.4.0
	 */
	public function test_get_object_ids_is_bounded_and_newest_first(): void {
		foreach ( array( 10, 30, 20, 40 ) as $id ) {
			$this->repository->save( $this->make_record( $id ) );
			$this->repository->save( $this->make_record( $id, array( 0.1, 0.2, 0.3 ), self::MODEL, 1 ) );
		}
		$this->repository->save( $this->make_record( 50, array( 0.1 ), 'other-model' ) );

		$this->assertSame( array( 40, 30, 20, 10 ), $this->repository->get_object_ids( 'post', self::PROVIDER, self::MODEL, 10 ) );
		$this->assertSame( array( 40, 30 ), $this->repository->get_object_ids( 'post', self::PROVIDER, self::MODEL, 2 ) );
		$this->assertSame( array( 20, 10 ), $this->repository->get_object_ids( 'post', self::PROVIDER, self::MODEL, 2, 2 ) );
		$this->assertSame( array(), $this->repository->get_object_ids( 'post', self::PROVIDER, self::MODEL, 0 ) );
		$this->assertSame( 4, $this->repository->count_objects( 'post', self::PROVIDER, self::MODEL ) );
	}

	/**
	 * Tests that iteration yields every record for a model across batches.
	 *
	 * @since 1.4.0
	 */
	public function test_iterate_yields_all_records_in_batches(): void {
		for ( $i = 1; $i <= 7; $i++ ) {
			$this->repository->save( $this->make_record( $i, array( $i / 10 ) ) );
		}
		$this->repository->save( $this->make_record( 99, array( 0.5 ), 'other-model' ) );
		$this->repository->save( $this->make_record( 100, array( 0.5 ), self::MODEL, 0, '', 'term' ) );

		$all = iterator_to_array( $this->repository->iterate( self::PROVIDER, self::MODEL, null, 3 ), false );
		$this->assertCount( 8, $all );
		$this->assertSame( range( 1, 7 ), array_map( static fn( Embedding_Record $r ): int => $r->get_object_id(), array_slice( $all, 0, 7 ) ) );

		$posts = iterator_to_array( $this->repository->iterate( self::PROVIDER, self::MODEL, 'post', 3 ), false );
		$this->assertCount( 7, $posts );

		$other = iterator_to_array( $this->repository->iterate( self::PROVIDER, 'other-model' ), false );
		$this->assertCount( 1, $other );
		$this->assertSame( 99, $other[0]->get_object_id() );
	}

	/**
	 * Tests deleting an object's vectors, optionally scoped to a model.
	 *
	 * @since 1.4.0
	 */
	public function test_delete_for_object(): void {
		$this->repository->save( $this->make_record( 1, array( 0.1 ), 'model-a' ) );
		$this->repository->save( $this->make_record( 1, array( 0.2 ), 'model-a', 1 ) );
		$this->repository->save( $this->make_record( 1, array( 0.3 ), 'model-b' ) );
		$this->repository->save( $this->make_record( 2, array( 0.4 ), 'model-a' ) );

		$this->assertSame( 2, $this->repository->delete_for_object( 'post', 1, self::PROVIDER, 'model-a' ) );
		$this->assertCount( 0, $this->repository->get( 'post', 1, self::PROVIDER, 'model-a' ) );
		$this->assertCount( 1, $this->repository->get( 'post', 1, self::PROVIDER, 'model-b' ) );

		$this->assertSame( 1, $this->repository->delete_for_object( 'post', 1 ) );
		$this->assertCount( 0, $this->repository->get( 'post', 1, self::PROVIDER, 'model-b' ) );
		$this->assertCount( 1, $this->repository->get( 'post', 2, self::PROVIDER, 'model-a' ) );
	}

	/**
	 * Tests deleting every vector produced by a model.
	 *
	 * @since 1.4.0
	 */
	public function test_delete_for_model(): void {
		$this->repository->save( $this->make_record( 1, array( 0.1 ), 'model-a' ) );
		$this->repository->save( $this->make_record( 2, array( 0.2 ), 'model-a' ) );
		$this->repository->save( $this->make_record( 3, array( 0.3 ), 'model-b' ) );

		$this->assertSame( 2, $this->repository->delete_for_model( self::PROVIDER, 'model-a' ) );
		$this->assertSame( 0, $this->repository->count_objects( 'post', self::PROVIDER, 'model-a' ) );
		$this->assertSame( 1, $this->repository->count_objects( 'post', self::PROVIDER, 'model-b' ) );
	}

	/**
	 * Tests that a row whose bytes no longer match its dimensions is skipped rather than fatal.
	 *
	 * @since 1.4.0
	 */
	public function test_corrupt_rows_are_skipped(): void {
		global $wpdb;

		$saved = $this->repository->save( $this->make_record( 1, array( 0.1, 0.2 ) ) );
		$this->repository->save( $this->make_record( 2, array( 0.3, 0.4 ) ) );

		$wpdb->update( $this->schema->get_table_name(), array( 'dimensions' => 5 ), array( 'id' => $saved->get_id() ), array( '%d' ), array( '%d' ) ); // phpcs:ignore WordPress.DB.DirectDatabaseQuery.DirectQuery,WordPress.DB.DirectDatabaseQuery.NoCaching

		$this->assertNull( $this->repository->get_by_id( $saved->get_id() ) );
		$this->assertCount( 1, iterator_to_array( $this->repository->iterate( self::PROVIDER, self::MODEL ), false ) );
	}

	/**
	 * Tests that a large, realistic vector round-trips.
	 *
	 * @since 1.4.0
	 */
	public function test_large_vector_round_trip(): void {
		$vector = array();
		for ( $i = 0; $i < 3072; $i++ ) {
			$vector[] = sin( $i ) / 10;
		}

		$this->repository->save( $this->make_record( 1, $vector, 'gemini-embedding-001', 0, '', 'post', 'google' ) );

		$records = $this->repository->get( 'post', 1, 'google', 'gemini-embedding-001' );

		$this->assertCount( 1, $records );
		$this->assertSame( 3072, $records[0]->get_dimensions() );
		$this->assertEqualsWithDelta( $vector, $records[0]->get_vector(), 1.0e-6 );
	}
}
