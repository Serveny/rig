use std::{f64::consts::FRAC_1_SQRT_2, io};

use rig::embeddings::{EmbedError, Embedding, EmbeddingError, EmbeddingModel, TextEmbedder};
use rig::vector_store::request::{SearchFilter, VectorSearchRequest};
use rig::vector_store::{InsertDocuments, VectorStoreIndex};
use rig::{Embed, OneOrMany};
use rig_surrealdb::{Mem, SurrealSearchFilter, SurrealVectorStore};
use serde::{Deserialize, Serialize};
use surrealdb::Surreal;
use surrealdb::engine::local::Db;
use surrealdb::types::Value;

#[derive(Clone, Debug, Deserialize, PartialEq, Eq, Serialize)]
struct TestDocument {
    title: String,
    category: String,
    body: String,
}

impl Embed for TestDocument {
    fn embed(&self, embedder: &mut TextEmbedder) -> Result<(), EmbedError> {
        embedder.embed(self.body.clone());
        Ok(())
    }
}

#[derive(Clone, Copy)]
struct DeterministicEmbeddingModel;

impl EmbeddingModel for DeterministicEmbeddingModel {
    const MAX_DOCUMENTS: usize = 1;

    type Client = ();

    fn make(_: &Self::Client, _: impl Into<String>, _: Option<usize>) -> Self {
        Self
    }

    fn ndims(&self) -> usize {
        2
    }

    async fn embed_texts(
        &self,
        texts: impl IntoIterator<Item = String> + rig::wasm_compat::WasmCompatSend,
    ) -> Result<Vec<Embedding>, EmbeddingError> {
        texts
            .into_iter()
            .map(|text| {
                let vec = match text.as_str() {
                    "fruit query" => vec![1.0, 0.0],
                    "vehicle query" => vec![0.0, 1.0],
                    _ => {
                        return Err(EmbeddingError::DocumentError(Box::new(io::Error::new(
                            io::ErrorKind::InvalidInput,
                            format!("unexpected query text: {text}"),
                        ))));
                    }
                };

                Ok(Embedding {
                    document: text,
                    vec,
                })
            })
            .collect()
    }
}

async fn setup_store() -> Result<SurrealVectorStore<Db, DeterministicEmbeddingModel>, anyhow::Error>
{
    let surreal = Surreal::new::<Mem>(()).await?;
    surreal.use_ns("test").use_db("test").await?;
    surreal
        .query(
            "DEFINE TABLE documents SCHEMAFULL;\
             DEFINE FIELD document ON TABLE documents TYPE object FLEXIBLE;\
             DEFINE FIELD embedding ON TABLE documents TYPE array<float>;\
             DEFINE FIELD embedded_text ON TABLE documents TYPE string;\
             DEFINE INDEX IF NOT EXISTS documents_embedding_vector_index ON documents \
             FIELDS embedding HNSW DIMENSION 2 DIST COSINE;",
        )
        .await?;

    Ok(SurrealVectorStore::with_defaults(
        DeterministicEmbeddingModel,
        surreal,
    ))
}

async fn insert_documents(
    store: &SurrealVectorStore<Db, DeterministicEmbeddingModel>,
) -> Result<(), anyhow::Error> {
    let documents = vec![
        (
            TestDocument {
                title: "apple".to_string(),
                category: "fruit".to_string(),
                body: "crisp and sweet".to_string(),
            },
            OneOrMany::one(Embedding {
                document: "crisp and sweet".to_string(),
                vec: vec![1.0, 0.0],
            }),
        ),
        (
            TestDocument {
                title: "banana".to_string(),
                category: "fruit".to_string(),
                body: "soft and yellow".to_string(),
            },
            OneOrMany::one(Embedding {
                document: "soft and yellow".to_string(),
                vec: vec![1.0, 1.0],
            }),
        ),
        (
            TestDocument {
                title: "sedan".to_string(),
                category: "vehicle".to_string(),
                body: "road vehicle".to_string(),
            },
            OneOrMany::one(Embedding {
                document: "road vehicle".to_string(),
                vec: vec![0.0, 1.0],
            }),
        ),
    ];

    store.insert_documents(documents).await?;
    Ok(())
}

#[tokio::test]
async fn top_n_returns_inserted_documents_in_distance_order() -> Result<(), anyhow::Error> {
    let store = setup_store().await?;
    insert_documents(&store).await?;

    let request = VectorSearchRequest::builder()
        .query("fruit query")
        .samples(3)
        .build()?;

    let results = store.top_n::<TestDocument>(request).await?;

    assert_eq!(results.len(), 3);
    assert_eq!(results[0].0, 0.0);
    assert_eq!(results[0].2.title, "apple");
    assert!((results[1].0 - (1.0 - FRAC_1_SQRT_2)).abs() < 1e-12);
    assert_eq!(results[1].2.title, "banana");
    assert_eq!(results[2].0, 1.0);
    assert_eq!(results[2].2.title, "sedan");

    Ok(())
}

#[tokio::test]
async fn top_n_ids_respects_thresholds() -> Result<(), anyhow::Error> {
    let store = setup_store().await?;
    insert_documents(&store).await?;

    let request = VectorSearchRequest::builder()
        .query("fruit query")
        .samples(3)
        .threshold(0.5)
        .build()?;

    let results = store.top_n_ids(request.clone()).await?;
    let full_results = store.top_n::<TestDocument>(request).await?;
    let expected_ids: Vec<_> = full_results.into_iter().map(|(_, id, _)| id).collect();

    assert_eq!(results.len(), 2);
    assert_eq!(results[0].0, 0.0);
    assert_eq!(results[0].1, expected_ids[0]);
    assert!((results[1].0 - (1.0 - FRAC_1_SQRT_2)).abs() < 1e-12);
    assert_eq!(results[1].1, expected_ids[1]);

    Ok(())
}

#[tokio::test]
async fn top_n_applies_filters_before_returning_results() -> Result<(), anyhow::Error> {
    let store = setup_store().await?;
    insert_documents(&store).await?;

    let request = VectorSearchRequest::builder()
        .query("vehicle query")
        .samples(3)
        .filter(SurrealSearchFilter::eq(
            "document.category",
            Value::String("fruit".to_string()),
        ))
        .build()?;

    let results = store.top_n::<TestDocument>(request).await?;

    assert_eq!(results.len(), 2);
    assert_eq!(results[0].2.title, "banana");
    assert_eq!(results[1].2.title, "apple");
    assert!(results.iter().all(|(_, _, doc)| doc.category == "fruit"));

    Ok(())
}
