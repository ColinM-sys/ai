<?php
/**
 * Binary encoding for embedding vectors.
 *
 * @package WordPress\AI\Embeddings
 */

declare( strict_types=1 );

namespace WordPress\AI\Embeddings;

use InvalidArgumentException;

defined( 'ABSPATH' ) || exit;

/**
 * Packs embedding vectors into compact little-endian float32 byte strings and back.
 *
 * Providers return vectors as float32 values, so storing them as float32 loses no
 * meaningful precision while using a quarter of the space of serialized PHP floats.
 * The encoding is the same one used by MariaDB's `VECTOR` type, which keeps a future
 * native backend able to read the same bytes.
 *
 * @since 1.4.0
 */
final class Vector_Codec {

	/**
	 * Number of bytes used per vector component.
	 */
	public const BYTES_PER_COMPONENT = 4;

	/**
	 * Packs a vector into a little-endian float32 byte string.
	 *
	 * @since 1.4.0
	 *
	 * @param list<int|float> $vector The vector to pack.
	 * @return string Packed bytes, `4 * count( $vector )` long.
	 *
	 * @throws \InvalidArgumentException If the vector is empty or contains non-finite values.
	 */
	public static function pack( array $vector ): string {
		self::validate( $vector );

		$packed = '';
		foreach ( $vector as $value ) {
			$packed .= pack( 'g', (float) $value );
		}

		return $packed;
	}

	/**
	 * Unpacks a little-endian float32 byte string into a vector.
	 *
	 * @since 1.4.0
	 *
	 * @param string $packed     Packed bytes as produced by {@see self::pack()}.
	 * @param int    $dimensions Expected number of components.
	 * @return list<float> The unpacked vector.
	 *
	 * @throws \InvalidArgumentException If the byte length does not match the expected dimensions.
	 */
	public static function unpack( string $packed, int $dimensions ): array {
		if ( $dimensions <= 0 || strlen( $packed ) !== $dimensions * self::BYTES_PER_COMPONENT ) {
			throw new InvalidArgumentException(
				esc_html(
					sprintf(
						'Packed vector is %d bytes, expected %d for %d dimensions.',
						strlen( $packed ),
						$dimensions * self::BYTES_PER_COMPONENT,
						$dimensions
					)
				)
			);
		}

		$values = unpack( 'g*', $packed );
		if ( ! is_array( $values ) || count( $values ) !== $dimensions ) {
			throw new InvalidArgumentException( 'Packed vector could not be decoded.' );
		}

		return array_map( 'floatval', array_values( $values ) );
	}

	/**
	 * Calculates the Euclidean (L2) norm of a vector.
	 *
	 * @since 1.4.0
	 *
	 * @param list<int|float> $vector The vector.
	 * @return float The norm.
	 */
	public static function norm( array $vector ): float {
		$sum = 0.0;
		foreach ( $vector as $value ) {
			$sum += (float) $value * (float) $value;
		}

		return sqrt( $sum );
	}

	/**
	 * Validates that a value is a non-empty list of finite numbers.
	 *
	 * @since 1.4.0
	 *
	 * @param mixed $vector The candidate vector.
	 * @return void
	 *
	 * @throws \InvalidArgumentException If the value is not a usable vector.
	 */
	public static function validate( $vector ): void {
		if ( ! is_array( $vector ) || array() === $vector || array_keys( $vector ) !== range( 0, count( $vector ) - 1 ) ) {
			throw new InvalidArgumentException( 'Embedding vector must be a non-empty list of numbers.' );
		}

		foreach ( $vector as $index => $value ) {
			if ( ! is_int( $value ) && ! is_float( $value ) ) {
				throw new InvalidArgumentException(
					esc_html( sprintf( 'Embedding vector component %d is not a number.', $index ) )
				);
			}

			if ( is_float( $value ) && ! is_finite( $value ) ) {
				throw new InvalidArgumentException(
					esc_html( sprintf( 'Embedding vector component %d is not finite.', $index ) )
				);
			}
		}
	}
}
