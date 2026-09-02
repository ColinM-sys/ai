<?php
/**
 * Tests for the embedding vector codec.
 *
 * @package WordPress\AI\Tests\Integration\Includes\Embeddings
 */

namespace WordPress\AI\Tests\Integration\Includes\Embeddings;

use InvalidArgumentException;
use WP_UnitTestCase;
use WordPress\AI\Embeddings\Vector_Codec;

/**
 * Vector_Codec test case.
 *
 * @since x.x.x
 *
 * @covers \WordPress\AI\Embeddings\Vector_Codec
 */
class Vector_CodecTest extends WP_UnitTestCase {

	/**
	 * Tests that packing produces four bytes per component.
	 *
	 * @since x.x.x
	 */
	public function test_pack_uses_four_bytes_per_component(): void {
		$packed = Vector_Codec::pack( array( 0.1, -0.2, 0.3 ) );

		$this->assertSame( 3 * Vector_Codec::BYTES_PER_COMPONENT, strlen( $packed ) );
	}

	/**
	 * Tests that packing is little-endian float32, the layout MariaDB's VECTOR type uses.
	 *
	 * @since x.x.x
	 */
	public function test_pack_is_little_endian_float32(): void {
		$this->assertSame( "\x00\x00\x80\x3f", Vector_Codec::pack( array( 1.0 ) ) );
		$this->assertSame( "\x00\x00\x00\xc0", Vector_Codec::pack( array( -2 ) ) );
	}

	/**
	 * Tests that a vector survives a round trip within float32 precision.
	 *
	 * @since x.x.x
	 */
	public function test_round_trip_preserves_values_within_float32_precision(): void {
		$vector = array( 0.123456789, -0.987654321, 42.0, 1.0e-5, 0 );

		$decoded = Vector_Codec::unpack( Vector_Codec::pack( $vector ), count( $vector ) );

		$this->assertCount( count( $vector ), $decoded );
		foreach ( $vector as $index => $value ) {
			$this->assertIsFloat( $decoded[ $index ] );
			$this->assertEqualsWithDelta( (float) $value, $decoded[ $index ], 1.0e-6 );
		}
	}

	/**
	 * Tests that unpacking rejects a byte string of the wrong length.
	 *
	 * @since x.x.x
	 */
	public function test_unpack_rejects_wrong_length(): void {
		$this->expectException( InvalidArgumentException::class );

		Vector_Codec::unpack( Vector_Codec::pack( array( 1.0, 2.0 ) ), 3 );
	}

	/**
	 * Tests that unpacking rejects non-positive dimensions.
	 *
	 * @since x.x.x
	 */
	public function test_unpack_rejects_zero_dimensions(): void {
		$this->expectException( InvalidArgumentException::class );

		Vector_Codec::unpack( '', 0 );
	}

	/**
	 * Tests the norm calculation.
	 *
	 * @since x.x.x
	 */
	public function test_norm(): void {
		$this->assertEqualsWithDelta( 5.0, Vector_Codec::norm( array( 3, 4 ) ), 1.0e-9 );
		$this->assertEqualsWithDelta( 0.0, Vector_Codec::norm( array( 0.0, 0.0 ) ), 1.0e-9 );
	}

	/**
	 * Tests that validation rejects values that are not usable vectors.
	 *
	 * @since x.x.x
	 *
	 * @dataProvider data_invalid_vectors
	 *
	 * @param mixed $vector The candidate vector.
	 */
	public function test_validate_rejects_invalid_vectors( $vector ): void {
		$this->expectException( InvalidArgumentException::class );

		Vector_Codec::validate( $vector );
	}

	/**
	 * Provides values that are not usable vectors.
	 *
	 * @since x.x.x
	 *
	 * @return array<string, array{0: mixed}> Test cases.
	 */
	public function data_invalid_vectors(): array {
		return array(
			'not an array'   => array( 'abc' ),
			'empty'          => array( array() ),
			'associative'    => array( array( 'x' => 1.0 ) ),
			'non-sequential' => array(
				array(
					1 => 1.0,
					2 => 2.0,
				),
			),
			'string value'   => array( array( 1.0, '2' ) ),
			'null value'     => array( array( 1.0, null ) ),
			'NAN'            => array( array( 1.0, NAN ) ),
			'INF'            => array( array( INF ) ),
		);
	}

	/**
	 * Tests that validation accepts integer and float components.
	 *
	 * @since x.x.x
	 */
	public function test_validate_accepts_numeric_list(): void {
		Vector_Codec::validate( array( 1, 2.5, -3 ) );

		$this->assertTrue( true );
	}

	/**
	 * Tests that values beyond float32 range are rejected rather than packed to INF.
	 *
	 * These are finite as PHP float64 values, so an `is_finite()` check passes them, but they
	 * round to `INF` on the way into four bytes. The write used to report success and the row was
	 * then unreadable on every subsequent read.
	 *
	 * @since x.x.x
	 *
	 * @dataProvider data_out_of_float32_range
	 *
	 * @param int|float $value The offending component.
	 */
	public function test_validate_rejects_values_outside_float32_range( $value ): void {
		$this->assertTrue( is_finite( (float) $value ), 'The fixture must be finite as a float64.' );

		$this->expectException( \InvalidArgumentException::class );
		$this->expectExceptionMessage( 'outside the range representable as float32' );

		Vector_Codec::validate( array( 0.1, $value ) );
	}

	/**
	 * Data provider for out-of-range components.
	 *
	 * @since x.x.x
	 *
	 * @return array<string, array{int|float}> Test cases.
	 */
	public function data_out_of_float32_range(): array {
		return array(
			'just above float32 max' => array( 3.5e38 ),
			'just below float32 min' => array( -3.5e38 ),
			'far above'              => array( 1.0e300 ),
			'far below'              => array( -1.0e300 ),
		);
	}

	/**
	 * Tests that the float32 boundary itself is still accepted and round-trips.
	 *
	 * @since x.x.x
	 */
	public function test_float32_boundary_is_accepted_and_round_trips(): void {
		$vector = array( Vector_Codec::MAX_MAGNITUDE, -Vector_Codec::MAX_MAGNITUDE );

		$unpacked = Vector_Codec::unpack( Vector_Codec::pack( $vector ), 2 );

		$this->assertTrue( is_finite( $unpacked[0] ), 'The boundary value must not pack to INF.' );
		$this->assertTrue( is_finite( $unpacked[1] ), 'The boundary value must not pack to INF.' );
	}

	/**
	 * Tests that a rejected vector never reaches the packed representation.
	 *
	 * @since x.x.x
	 */
	public function test_pack_rejects_values_outside_float32_range(): void {
		$this->expectException( \InvalidArgumentException::class );

		Vector_Codec::pack( array( 1.0e300 ) );
	}
}
