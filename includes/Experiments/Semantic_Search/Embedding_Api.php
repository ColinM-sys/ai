<?php
/**
 * Embedding API — HTTP wrapper for multiple providers.
 *
 * Reads configuration from the namespaced experiment options set via the AI
 * plugin's settings page.
 *
 * TODO: Replace this class entirely once WordPress/php-ai-client#244 lands.
 *       At that point, call wp_ai_client_prompt()->generateEmbeddingResult()
 *       and let the registered connector handle authentication + transport.
 *
 * @package WordPress\AI\Experiments\Semantic_Search
 */

declare( strict_types=1 );

namespace WordPress\AI\Experiments\Semantic_Search;

// Exit if accessed directly.
defined( 'ABSPATH' ) || exit;

/**
 * Sends text to an embedding API and returns a float vector.
 *
 * Supports four providers out of the box: OpenAI, Jina AI, Ollama (local),
 * and Google Gemini. Provider selection and credentials are read from the
 * namespaced WordPress options registered by the Semantic_Search experiment.
 *
 * @internal
 * @since x.x.x
 */
class Embedding_Api {

	/**
	 * Per-provider configuration defaults.
	 *
	 * Each entry contains:
	 *   - label:           Human-readable provider name.
	 *   - default_url:     API base URL used when no override is saved.
	 *   - default_model:   Model identifier used when no override is saved.
	 *   - dimensions:      Output vector length. Source: official model documentation.
	 *   - score_threshold: Cosine similarity cut-off. These are heuristics — no
	 *                      authoritative published benchmarks exist. Tune via Settings.
	 *
	 * @since x.x.x
	 * @var array<string, array{label:string, default_url:string, default_model:string, dimensions:int, score_threshold:float}>
	 */
	public const PROVIDERS = array(
		'openai' => array(
			'label'           => 'OpenAI',
			'default_url'     => 'https://api.openai.com/v1/embeddings',
			'default_model'   => 'text-embedding-3-small',
			'dimensions'      => 1536,
			'score_threshold' => 0.60,
		),
		'jina'   => array(
			'label'           => 'Jina AI',
			'default_url'     => 'https://api.jina.ai/v1/embeddings',
			'default_model'   => 'jina-embeddings-v3',
			'dimensions'      => 1024,
			'score_threshold' => 0.55,
		),
		'ollama' => array(
			'label'           => 'Ollama (local)',
			'default_url'     => 'http://localhost:11434/v1/embeddings',
			'default_model'   => 'nomic-embed-text',
			'dimensions'      => 768,
			'score_threshold' => 0.50,
		),
		'google' => array(
			'label'           => 'Google Gemini',
			'default_url'     => 'https://generativelanguage.googleapis.com/v1/models',
			'default_model'   => 'gemini-embedding-001',
			'dimensions'      => 3072,
			'score_threshold' => 0.45,
		),
	);

	/**
	 * Active provider key (e.g. 'openai', 'google').
	 *
	 * @since x.x.x
	 * @var string
	 */
	private string $provider;

	/**
	 * API key for the active provider. Empty for Ollama.
	 *
	 * @since x.x.x
	 * @var string
	 */
	private string $api_key;

	/**
	 * Embedding model identifier (e.g. 'text-embedding-3-small').
	 *
	 * @since x.x.x
	 * @var string
	 */
	private string $model;

	/**
	 * API base URL for the active provider.
	 *
	 * @since x.x.x
	 * @var string
	 */
	private string $base_url;

	/**
	 * Human-readable description of the last API error, or empty string if none.
	 *
	 * @since x.x.x
	 * @var string
	 */
	private string $last_error = '';

	/**
	 * Reads all provider configuration from the saved experiment options.
	 *
	 * @since x.x.x
	 */
	public function __construct() {
		$this->provider = (string) get_option( Semantic_Search::get_field_option_name( 'provider' ), 'openai' );
		$this->api_key  = (string) get_option( Semantic_Search::get_field_option_name( 'api_key' ), '' );
		$this->model    = (string) get_option( Semantic_Search::get_field_option_name( 'model' ), 'text-embedding-3-small' );
		$this->base_url = (string) get_option( Semantic_Search::get_field_option_name( 'base_url' ), 'https://api.openai.com/v1/embeddings' );
	}

	/**
	 * Returns the active model identifier.
	 *
	 * @since x.x.x
	 *
	 * @return string Model identifier string (e.g. 'gemini-embedding-001').
	 */
	public function get_model(): string {
		return $this->model;
	}

	/**
	 * Returns the active provider key.
	 *
	 * @since x.x.x
	 *
	 * @return string Provider key (e.g. 'openai', 'google').
	 */
	public function get_provider(): string {
		return $this->provider;
	}

	/**
	 * Returns the human-readable error message from the most recent API call.
	 *
	 * Returns an empty string when the last call succeeded.
	 *
	 * @since x.x.x
	 *
	 * @return string Error description, or empty string on success.
	 */
	public function get_last_error(): string {
		return $this->last_error;
	}

	/**
	 * Returns whether the provider is ready to accept embedding requests.
	 *
	 * Ollama requires only a non-empty model name (no key). All other providers
	 * require a non-empty API key.
	 *
	 * @since x.x.x
	 *
	 * @return bool True when the minimum required credentials are present.
	 */
	public function is_configured(): bool {
		if ( 'ollama' === $this->provider ) {
			return '' !== $this->model;
		}

		return '' !== $this->api_key;
	}

	/**
	 * Returns the cosine similarity score threshold for the active provider.
	 *
	 * Reads the user-saved option first; falls back to the per-provider default
	 * from PROVIDERS when the option is empty. An empty string means "use the
	 * provider default" and is backward-compatible with existing installs that
	 * predate the user-editable threshold field.
	 *
	 * @since x.x.x
	 *
	 * @return float Cosine similarity cut-off in the range [0, 1].
	 */
	public function get_score_threshold(): float {
		$cfg   = self::PROVIDERS[ $this->provider ] ?? self::PROVIDERS['openai'];
		$saved = (string) get_option( Semantic_Search::get_field_option_name( 'score_threshold' ), '' );

		return '' !== $saved ? (float) $saved : $cfg['score_threshold'];
	}

	/**
	 * Sends text to the configured embedding provider and returns a float vector.
	 *
	 * Dispatches to the Google-specific or OpenAI-compatible implementation
	 * depending on the active provider. Returns null on any API error; call
	 * get_last_error() to retrieve the reason.
	 *
	 * @since x.x.x
	 *
	 * @param string $text The text to embed.
	 * @return float[]|null Float vector on success, null on failure.
	 */
	public function generate( string $text ): ?array {
		$this->last_error = '';

		if ( 'google' === $this->provider ) {
			return $this->generate_google( $text );
		}

		return $this->generate_openai_compatible( $text );
	}

	/**
	 * Generates an embedding using the OpenAI-compatible request format.
	 *
	 * Used for OpenAI, Jina AI, and Ollama. The Authorization header is omitted
	 * for Ollama since it requires no API key.
	 *
	 * @since x.x.x
	 *
	 * @param string $text The text to embed.
	 * @return float[]|null Float vector on success, null on failure.
	 */
	private function generate_openai_compatible( string $text ): ?array {
		$headers = array( 'Content-Type' => 'application/json' );

		if ( 'ollama' !== $this->provider ) {
			$headers['Authorization'] = 'Bearer ' . $this->api_key;
		}

		$response = wp_remote_post(
			$this->base_url,
			array(
				'timeout' => 30,
				'headers' => $headers,
				'body'    => wp_json_encode(
					array(
						'input' => $text,
						'model' => $this->model,
					)
				),
			)
		);

		return $this->parse_openai_response( $response );
	}

	/**
	 * Generates an embedding using the Google Gemini embedContent API.
	 *
	 * Google's API differs from the OpenAI-compatible format in three ways:
	 *   - The model name is appended to the URL path, not sent in the body.
	 *   - The API key is passed as a query parameter, not an Authorization header.
	 *   - The request body uses `content.parts[].text` instead of `input`.
	 *
	 * @since x.x.x
	 *
	 * @param string $text The text to embed.
	 * @return float[]|null Float vector on success, null on failure.
	 */
	private function generate_google( string $text ): ?array {
		$endpoint = rtrim( $this->base_url, '/' ) . '/' . $this->model . ':embedContent?key=' . $this->api_key;

		$response = wp_remote_post(
			$endpoint,
			array(
				'timeout' => 30,
				'headers' => array( 'Content-Type' => 'application/json' ),
				'body'    => wp_json_encode(
					array(
						'content' => array(
							'parts' => array(
								array( 'text' => $text ),
							),
						),
					)
				),
			)
		);

		if ( is_wp_error( $response ) ) {
			$this->last_error = $response->get_error_message();
			return null;
		}

		$code = wp_remote_retrieve_response_code( $response );
		$body = json_decode( wp_remote_retrieve_body( $response ), true );

		if ( 200 !== $code ) {
			$this->last_error = sprintf( 'HTTP %d: %s', $code, $body['error']['message'] ?? 'unknown error' );
			return null;
		}

		$values = $body['embedding']['values'] ?? null;

		if ( ! is_array( $values ) ) {
			$this->last_error = 'Unexpected response format from Google API.';
			return null;
		}

		return array_map( 'floatval', $values );
	}

	/**
	 * Parses an OpenAI-compatible embedding response into a float vector.
	 *
	 * Handles WP_Error from wp_remote_post, non-200 HTTP status codes, and
	 * unexpected response shapes. On any failure, sets last_error and returns null.
	 *
	 * @since x.x.x
	 *
	 * @param \WP_Error|array<string,mixed> $response Response from wp_remote_post().
	 * @return float[]|null Float vector on success, null on failure.
	 */
	private function parse_openai_response( $response ): ?array {
		if ( is_wp_error( $response ) ) {
			$this->last_error = $response->get_error_message();
			return null;
		}

		$code = wp_remote_retrieve_response_code( $response );
		$body = json_decode( wp_remote_retrieve_body( $response ), true );

		if ( 200 !== $code ) {
			$this->last_error = sprintf( 'HTTP %d: %s', $code, $body['error']['message'] ?? 'unknown error' );
			return null;
		}

		$values = $body['data'][0]['embedding'] ?? null;

		if ( ! is_array( $values ) ) {
			$this->last_error = 'Unexpected response format from provider.';
			return null;
		}

		return array_map( 'floatval', $values );
	}

	/**
	 * Fetches the names of embedding-capable models available for the current Google API key.
	 *
	 * Calls the Google ListModels endpoint and filters to models that support the
	 * `embedContent` method. Used only by the test-connection REST endpoint to
	 * surface actionable context when the configured model name returns a 404.
	 *
	 * @since x.x.x
	 *
	 * @return string[] Model name strings (e.g. ['gemini-embedding-001']), empty on error.
	 */
	public function list_google_embedding_models(): array {
		$endpoint = 'https://generativelanguage.googleapis.com/v1/models?key=' . $this->api_key;
		$response = wp_remote_get( $endpoint, array( 'timeout' => 15 ) );

		if ( is_wp_error( $response ) ) {
			return array();
		}

		$body   = json_decode( wp_remote_retrieve_body( $response ), true );
		$models = $body['models'] ?? array();
		$names  = array();

		foreach ( $models as $m ) {
			$supported = $m['supportedGenerationMethods'] ?? array();
			if ( in_array( 'embedContent', $supported, true ) ) {
				$names[] = str_replace( 'models/', '', $m['name'] ?? '' );
			}
		}

		return array_filter( $names );
	}
}
