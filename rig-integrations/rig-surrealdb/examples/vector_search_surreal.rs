use rig::client::{EmbeddingsClient, ProviderClient};
use rig::providers::openai;
use rig::vector_store::request::VectorSearchRequest;
use rig::{
    Embed,
    embeddings::EmbeddingsBuilder,
    vector_store::{InsertDocuments, VectorStoreIndex},
};
use rig_surrealdb::{Mem, SurrealSearchFilter, SurrealVectorStore};
use serde::{Deserialize, Serialize};
use surrealdb::Surreal;

// A vector search needs to be performed on the `definitions` field, so we derive the `Embed` trait for `WordDefinition`
// and tag that field with `#[embed]`.
// We are not going to store the definitions on our database so we skip the `Serialize` trait
#[derive(Embed, Serialize, Deserialize, Clone, Debug, Eq, PartialEq, Default)]
struct WordDefinition {
    word: String,
    #[serde(skip)] // we don't want to serialize this field, we use only to create embeddings
    #[embed]
    definition: String,
}

impl std::fmt::Display for WordDefinition {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.word)
    }
}

fn threshold_request(
    query: &str,
    samples: u64,
    threshold: f64,
) -> Result<VectorSearchRequest<SurrealSearchFilter>, rig::vector_store::VectorStoreError> {
    VectorSearchRequest::builder()
        .query(query)
        .samples(samples)
        .threshold(threshold)
        .build()
}

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // Create OpenAI client
    let openai_client = openai::Client::from_env();
    let model = openai_client.embedding_model(openai::TEXT_EMBEDDING_ADA_002);

    let surreal = Surreal::new::<Mem>(()).await?;
    surreal.use_ns("example").use_db("example").await?;

    // Keep the in-memory example self-contained by defining the schema inline.
    // For a persistent SurrealDB 3 deployment, apply `examples/migrations.surql`
    // ahead of time instead of creating schema during application startup.
    surreal
        .query(
            "DEFINE TABLE documents SCHEMAFULL;\
             DEFINE FIELD document ON TABLE documents TYPE object FLEXIBLE;\
             DEFINE FIELD embedding ON TABLE documents TYPE array<float>;\
             DEFINE FIELD embedded_text ON TABLE documents TYPE string;\
             DEFINE INDEX IF NOT EXISTS words_embedding_vector_index ON documents \
             FIELDS embedding HNSW DIMENSION 1536 DIST COSINE;",
        )
        .await?;

    // create test documents with mocked embeddings
    let words = vec![
        WordDefinition {
            word: "flurbo".to_string(),
            definition: "1. *flurbo* (name): A fictional digital currency that originated in the animated series Rick and Morty.".to_string()
        },
        WordDefinition {
            word: "glarb-glarb".to_string(),
            definition: "1. *glarb-glarb* (noun): A fictional creature found in the distant, swampy marshlands of the planet Glibbo in the Andromeda galaxy.".to_string()
        },
        WordDefinition {
            word: "linglingdong".to_string(),
            definition: "1. *linglingdong* (noun): A term used by inhabitants of the far side of the moon to describe humans.".to_string(),
        }];

    let documents = EmbeddingsBuilder::new(model.clone())
        .documents(words)?
        .build()
        .await?;

    // init vector store
    let vector_store = SurrealVectorStore::with_defaults(model, surreal);

    vector_store.insert_documents(documents).await?;

    // query vector
    let query = "What does \"glarb-glarb\" mean?";
    println!("Attempting vector search with query: {query}");

    let req = VectorSearchRequest::builder()
        .query(query)
        .samples(2)
        .build()?;

    let results = vector_store.top_n::<WordDefinition>(req).await?;

    println!("{} results for query: {}", results.len(), query);
    for (distance, _id, doc) in results.iter() {
        println!("Result distance {distance} for word: {doc}");
    }

    let req = VectorSearchRequest::builder()
        .query(query)
        .samples(2)
        .build()?;

    let id_results = vector_store.top_n_ids(req).await?;
    println!("Top result ids: {id_results:?}");

    // This fixed threshold only demonstrates the API shape; live model distances vary.
    let threshold = 0.25;

    println!(
        "Attempting vector search with cosine distance threshold of {threshold} and query: {query}"
    );
    let req = threshold_request(query, 2, threshold)?;

    let results = vector_store.top_n::<WordDefinition>(req).await?;

    println!("{} results for query: {}", results.len(), query);

    for (distance, _id, doc) in results.iter() {
        println!("Result distance {distance} for word: {doc}");
    }

    Ok(())
}
