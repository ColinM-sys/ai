<?php
/**
 * Tests for the embeddings table schema.
 *
 * @package WordPress\AI\Tests\Integration\Includes\Embeddings
 */

namespace WordPress\AI\Tests\Integration\Includes\Embeddings;

use WP_UnitTestCase;
use WordPress\AI\Embeddings\Embedding_Schema;

/**
 * Embedding_Schema test case.
 *
 * @since x.x.x
 *
 * @covers \WordPress\AI\Embeddings\Embedding_Schema
 */
class Embedding_SchemaTest extends WP_UnitTestCase {

	/**
	 * Schema under test.
	 *
	 * @var \WordPress\AI\Embeddings\Embedding_Schema
	 */
	private Embedding_Schema $schema;

	/**
	 * Set up test case.
	 *
	 * @since x.x.x
	 */
	protected function setUp(): void {
		parent::setUp();

		$this->schema = new Embedding_Schema();
		$this->schema->drop_table();
	}

	/**
	 * Tear down test case.
	 *
	 * @since x.x.x
	 */
	protected function tearDown(): void {
		$this->schema->drop_table();

		parent::tearDown();
	}

	/**
	 * Tests the prefixed table name.
	 *
	 * @since x.x.x
	 */
	public function test_get_table_name_is_prefixed(): void {
		global $wpdb;

		$this->assertSame( $wpdb->prefix . 'wpai_embeddings', $this->schema->get_table_name() );
	}

	/**
	 * Tests that upgrading creates the table and records the schema version.
	 *
	 * @since x.x.x
	 */
	public function test_maybe_upgrade_table_creates_table_and_records_version(): void {
		$this->assertFalse( $this->schema->table_exists() );

		$this->schema->maybe_upgrade_table();

		$this->assertTrue( $this->schema->table_exists() );
		$this->assertSame( '1', get_option( Embedding_Schema::SCHEMA_VERSION_OPTION ) );
	}

	/**
	 * Tests that the table has the expected columns and unique key.
	 *
	 * @since x.x.x
	 */
	public function test_table_has_expected_columns_and_unique_key(): void {
		global $wpdb;

		$this->schema->maybe_upgrade_table();

		$table   = $this->schema->get_table_name();
		$columns = $wpdb->get_col( "SHOW COLUMNS FROM {$table}" ); // phpcs:ignore WordPress.DB.DirectDatabaseQuery.DirectQuery,WordPress.DB.DirectDatabaseQuery.NoCaching,WordPress.DB.PreparedSQL.InterpolatedNotPrepared

		$this->assertSame(
			array( 'id', 'object_type', 'object_id', 'chunk_index', 'provider', 'model', 'dimensions', 'embedding', 'embedding_norm', 'content_hash', 'created_at', 'updated_at' ),
			$columns
		);

		$indexes = $wpdb->get_results( "SHOW INDEX FROM {$table}", ARRAY_A ); // phpcs:ignore WordPress.DB.DirectDatabaseQuery.DirectQuery,WordPress.DB.DirectDatabaseQuery.NoCaching,WordPress.DB.PreparedSQL.InterpolatedNotPrepared
		$names   = array_unique( array_column( $indexes, 'Key_name' ) );

		$this->assertContains( 'uniq_object_model_chunk', $names );
		$this->assertContains( 'idx_provider_model', $names );
		$this->assertContains( 'idx_object', $names );
	}

	/**
	 * Tests that upgrading is a no-op once the table exists at the current version.
	 *
	 * @since x.x.x
	 */
	public function test_maybe_upgrade_table_is_idempotent(): void {
		$this->schema->maybe_upgrade_table();
		$this->schema->maybe_upgrade_table();

		$this->assertTrue( $this->schema->table_exists() );
	}

	/**
	 * Tests that a stale version option does not stop the table from being recreated.
	 *
	 * @since x.x.x
	 */
	public function test_maybe_upgrade_table_recreates_missing_table_despite_version_option(): void {
		update_option( Embedding_Schema::SCHEMA_VERSION_OPTION, '1', false );

		$this->schema->maybe_upgrade_table();

		$this->assertTrue( $this->schema->table_exists() );
	}

	/**
	 * Tests that dropping removes the table and the version option.
	 *
	 * @since x.x.x
	 */
	public function test_drop_table(): void {
		$this->schema->maybe_upgrade_table();
		$this->schema->drop_table();

		$this->assertFalse( $this->schema->table_exists() );
		$this->assertFalse( get_option( Embedding_Schema::SCHEMA_VERSION_OPTION ) );
	}
}
