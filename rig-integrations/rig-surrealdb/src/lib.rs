//! SurrealDB-backed vector search integration for Rig.

use std::fmt::Display;

use rig::{
    Embed, OneOrMany,
    embeddings::{Embedding, EmbeddingModel},
    vector_store::{
        InsertDocuments, VectorStoreError, VectorStoreIndex,
        request::{SearchFilter, VectorSearchRequest},
    },
    wasm_compat::{WasmCompatSend, WasmCompatSync},
};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use surrealdb::{
    Connection, Surreal,
    types::{RecordId, SurrealValue, ToSql, Value},
};

pub use surrealdb::engine::local::Mem;
pub use surrealdb::engine::remote::ws::{Ws, Wss};

/// A Rig vector store backed by SurrealDB records and vector indexes.
pub struct SurrealVectorStore<C, Model>
where
    C: Connection + WasmCompatSend + WasmCompatSync,
    Model: EmbeddingModel,
{
    model: Model,
    surreal: Surreal<C>,
    documents_table: String,
    distance_function: SurrealDistanceFunction,
}

/// Selects the query semantics used by `SurrealVectorStore`.
///
/// In SurrealDB v3, `Knn`, `Cosine`, `Euclidean`, and `Hamming` all use the
/// same index-backed HNSW query path. For those variants, the effective metric
/// comes from the vector index `DIST` setting, so the index definition must
/// match the chosen variant.
///
/// `Jaccard` remains a brute-force similarity query and does not use the
/// index-backed KNN path.
pub enum SurrealDistanceFunction {
    /// Uses SurrealDB's generic index-backed KNN path.
    ///
    /// The score semantics follow the vector index `DIST` setting.
    Knn,
    /// Uses the index-backed KNN path and expects the vector index to use
    /// `DIST HAMMING`.
    Hamming,
    /// Uses the index-backed KNN path and expects the vector index to use
    /// `DIST EUCLIDEAN`.
    Euclidean,
    /// Uses the index-backed KNN path and expects the vector index to use
    /// `DIST COSINE`.
    Cosine,
    /// Uses `vector::similarity::jaccard($vec, embedding)` directly instead of
    /// the index-backed KNN path.
    Jaccard,
}

impl Display for SurrealDistanceFunction {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            SurrealDistanceFunction::Cosine => write!(f, "cosine"),
            SurrealDistanceFunction::Knn => write!(f, "knn"),
            SurrealDistanceFunction::Euclidean => write!(f, "euclidean"),
            SurrealDistanceFunction::Hamming => write!(f, "hamming"),
            SurrealDistanceFunction::Jaccard => write!(f, "jaccard"),
        }
    }
}

impl SurrealDistanceFunction {
    #[cfg(test)]
    fn uses_index_distance(&self) -> bool {
        !matches!(self, SurrealDistanceFunction::Jaccard)
    }

    fn select_distance_expression(&self) -> &'static str {
        match self {
            SurrealDistanceFunction::Jaccard => "vector::similarity::jaccard($vec, embedding)",
            _ => "vector::distance::knn()",
        }
    }

    fn knn_clause(&self, samples: u64) -> String {
        match self {
            SurrealDistanceFunction::Knn
            | SurrealDistanceFunction::Cosine
            | SurrealDistanceFunction::Euclidean
            | SurrealDistanceFunction::Hamming => {
                format!("embedding <|{samples},{samples}|> $vec")
            }
            // Jaccard is the only variant whose query semantics do not come from
            // the vector index DIST setting, so it stays on the direct similarity path.
            SurrealDistanceFunction::Jaccard => "true".to_string(),
        }
    }

    fn threshold_operator(&self) -> &'static str {
        match self {
            SurrealDistanceFunction::Jaccard => ">=",
            _ => "<=",
        }
    }

    fn sort_direction(&self) -> &'static str {
        match self {
            SurrealDistanceFunction::Jaccard => "DESC",
            _ => "ASC",
        }
    }
}

#[derive(Debug, Deserialize, SurrealValue)]
#[surreal(crate = "surrealdb::types")]
struct SearchResult {
    id: RecordId,
    document: serde_json::Value,
    distance: f64,
}

/// The SurrealDB row shape inserted by [`InsertDocuments`].
#[derive(Debug, Serialize, Deserialize, SurrealValue)]
#[surreal(crate = "surrealdb::types")]
pub struct CreateRecord {
    document: serde_json::Value,
    embedded_text: String,
    embedding: Vec<f64>,
}

/// The SurrealDB row shape returned by id-only vector searches.
#[derive(Debug, Deserialize, SurrealValue)]
#[surreal(crate = "surrealdb::types")]
pub struct SearchResultOnlyId {
    id: RecordId,
    distance: f64,
}

impl SearchResult {
    pub fn into_result<T: DeserializeOwned>(self) -> Result<(f64, String, T), VectorStoreError> {
        let document: T =
            serde_json::from_value(self.document).map_err(VectorStoreError::JsonError)?;

        Ok((self.distance, self.id.to_sql(), document))
    }
}

impl<C, Model> InsertDocuments for SurrealVectorStore<C, Model>
where
    C: Connection + WasmCompatSend + WasmCompatSync,
    Model: EmbeddingModel,
{
    async fn insert_documents<Doc: Serialize + Embed + WasmCompatSend>(
        &self,
        documents: Vec<(Doc, OneOrMany<Embedding>)>,
    ) -> Result<(), VectorStoreError> {
        for (document, embeddings) in documents {
            let json_document = serde_json::to_value(&document)?;

            for embedding in embeddings {
                let embedded_text = embedding.document;
                let embedding: Vec<f64> = embedding.vec;

                let record = CreateRecord {
                    document: json_document.clone(),
                    embedded_text,
                    embedding,
                };

                self.surreal
                    .create::<Option<CreateRecord>>(self.documents_table.clone())
                    .content(record)
                    .await
                    .map_err(|e| VectorStoreError::DatastoreError(Box::new(e)))?;
            }
        }

        Ok(())
    }
}

/// A SurrealDB-native filter expression used in vector search queries.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SurrealSearchFilter(String);

impl SurrealSearchFilter {
    fn inner(self) -> String {
        self.0
    }
}

impl std::fmt::Display for SurrealSearchFilter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl SearchFilter for SurrealSearchFilter {
    type Value = Value;

    fn eq(key: impl AsRef<str>, value: Self::Value) -> Self {
        Self(format!("{} = {}", key.as_ref(), value.to_sql()))
    }

    fn gt(key: impl AsRef<str>, value: Self::Value) -> Self {
        Self(format!("{} > {}", key.as_ref(), value.to_sql()))
    }

    fn lt(key: impl AsRef<str>, value: Self::Value) -> Self {
        Self(format!("{} < {}", key.as_ref(), value.to_sql()))
    }

    fn and(self, rhs: Self) -> Self {
        Self(format!("({self}) AND ({rhs})"))
    }

    fn or(self, rhs: Self) -> Self {
        Self(format!("({self}) OR ({rhs})"))
    }
}

impl SurrealSearchFilter {
    #[allow(clippy::should_implement_trait)]
    pub fn not(self) -> Self {
        Self(format!("NOT ({self})"))
    }

    /// Test if the value at `key` contains `val`
    pub fn contains(key: String, val: <Self as SearchFilter>::Value) -> Self {
        Self(format!("{key} CONTAINS {}", val.to_sql()))
    }

    /// Test if the value at `key` does *not* contain `val`
    pub fn does_not_contain(key: String, val: <Self as SearchFilter>::Value) -> Self {
        Self(format!("{key} CONTAINSNOT {}", val.to_sql()))
    }

    /// Test if the value at `key` contains every element of `vals`
    /// `vals` should be a SurrealDB collection
    pub fn all(key: String, vals: <Self as SearchFilter>::Value) -> Self {
        Self(format!("{key} CONTAINSALL {}", vals.to_sql()))
    }

    /// Test if the value at `key` contains any elements of `vals`
    /// `vals` should be a SurrealDB collection
    pub fn any(key: String, vals: <Self as SearchFilter>::Value) -> Self {
        Self(format!("{key} CONTAINSANY {}", vals.to_sql()))
    }

    /// Test if the value at `key` is a member of `vals`
    /// `vals` should be a SurrealDB collection
    pub fn member(key: String, vals: <Self as SearchFilter>::Value) -> Self {
        Self(format!("{key} IN {}", vals.to_sql()))
    }

    /// Test if the value at `key` is *not* a member of `vals`
    /// `vals` should be a SurrealDB collection
    pub fn not_member(key: String, vals: <Self as SearchFilter>::Value) -> Self {
        Self(format!("{key} NOTIN {}", vals.to_sql()))
    }

    // Geospatial filters
    /// Test if the value at `key` is inside `geometry`
    pub fn inside(key: String, geometry: <Self as SearchFilter>::Value) -> Self {
        Self(format!("{key} INSIDE {}", geometry.to_sql()))
    }

    /// Test if the value at `key` is outside `geometry`
    pub fn outside(key: String, geometry: <Self as SearchFilter>::Value) -> Self {
        Self(format!("{key} OUTSIDE {}", geometry.to_sql()))
    }

    /// Test if the value at `key` intersects `geometry`
    pub fn intersects(key: String, geometry: <Self as SearchFilter>::Value) -> Self {
        Self(format!("{key} INTERSECTS {}", geometry.to_sql()))
    }

    // String ops
    /// SurrealDB text search
    pub fn matches<'a, S: AsRef<&'a str>>(key: String, query: S) -> Self {
        Self(format!("{key} @@ {}", query.as_ref()))
    }

    /// Check if the value at `key` matches regex `pattern`
    /// `pattern` should be a valid surrealDB regex
    pub fn regex<'a, S: AsRef<&'a str>>(key: String, pattern: S) -> Self {
        Self(format!("{key} = /{}/", pattern.as_ref()))
    }
}

impl<C, Model> SurrealVectorStore<C, Model>
where
    C: Connection + WasmCompatSend + WasmCompatSync,
    Model: EmbeddingModel,
{
    /// Creates a vector store for the given SurrealDB client, table, and distance mode.
    pub fn new(
        model: Model,
        surreal: Surreal<C>,
        documents_table: Option<String>,
        distance_function: SurrealDistanceFunction,
    ) -> Self {
        Self {
            model,
            surreal,
            documents_table: documents_table.unwrap_or(String::from("documents")),
            distance_function,
        }
    }

    /// Returns the underlying SurrealDB client for schema management or custom queries.
    pub fn inner_client(&self) -> &Surreal<C> {
        &self.surreal
    }

    /// Creates a store targeting the default `documents` table with cosine distance semantics.
    pub fn with_defaults(model: Model, surreal: Surreal<C>) -> Self {
        Self::new(model, surreal, None, SurrealDistanceFunction::Cosine)
    }

    fn search_query(
        &self,
        with_document: bool,
        samples: u64,
        filter: &str,
        has_threshold: bool,
    ) -> String {
        let fields = if with_document { "id, document" } else { "id" };
        let knn_clause = self.distance_function.knn_clause(samples);
        let where_clause = match filter {
            "" | "true" => knn_clause,
            _ if knn_clause == "true" => filter.to_string(),
            _ => format!("{filter} AND {knn_clause}"),
        };
        let inner_query = format!(
            "SELECT {fields}, {} AS distance FROM type::table($tablename) WHERE {where_clause} ORDER BY distance {} LIMIT {samples}",
            self.distance_function.select_distance_expression(),
            self.distance_function.sort_direction(),
        );

        if has_threshold {
            format!(
                "SELECT {fields}, distance FROM ({inner_query}) WHERE distance {} $threshold ORDER BY distance {} LIMIT {samples}",
                self.distance_function.threshold_operator(),
                self.distance_function.sort_direction(),
            )
        } else {
            inner_query
        }
    }

    fn filter_expression(req: &VectorSearchRequest<SurrealSearchFilter>) -> String {
        req.filter()
            .clone()
            .map(SurrealSearchFilter::inner)
            .unwrap_or_else(|| "true".to_string())
    }
}

impl<C, Model> VectorStoreIndex for SurrealVectorStore<C, Model>
where
    C: Connection + WasmCompatSend + WasmCompatSync,
    Model: EmbeddingModel,
{
    type Filter = SurrealSearchFilter;

    /// Get the top n documents based on the distance to the given query.
    /// The result is a list of tuples of the form (score, id, document)
    async fn top_n<T: for<'a> Deserialize<'a> + WasmCompatSend>(
        &self,
        req: VectorSearchRequest<SurrealSearchFilter>,
    ) -> Result<Vec<(f64, String, T)>, VectorStoreError> {
        if req.samples() == 0 {
            return Ok(Vec::new());
        }

        let embedded_query: Vec<f64> = self.model.embed_text(req.query()).await?.vec;
        let filter = Self::filter_expression(&req);
        let query = self.search_query(true, req.samples(), &filter, req.threshold().is_some());

        let mut response = self
            .surreal
            .query(query.as_str())
            .bind(("vec", embedded_query))
            .bind(("tablename", self.documents_table.clone()))
            .bind(("threshold", req.threshold().unwrap_or(0.)))
            .await
            .map_err(|e| VectorStoreError::DatastoreError(Box::new(e)))?;

        let rows: Vec<SearchResult> = response
            .take(0)
            .map_err(|e| VectorStoreError::DatastoreError(Box::new(e)))?;

        let rows: Vec<(f64, String, T)> = rows
            .into_iter()
            .map(SearchResult::into_result)
            .collect::<Result<_, _>>()?;

        Ok(rows)
    }

    /// Same as `top_n` but returns the document ids only.
    async fn top_n_ids(
        &self,
        req: VectorSearchRequest<SurrealSearchFilter>,
    ) -> Result<Vec<(f64, String)>, VectorStoreError> {
        if req.samples() == 0 {
            return Ok(Vec::new());
        }

        let embedded_query: Vec<f64> = self.model.embed_text(req.query()).await?.vec;
        let filter = Self::filter_expression(&req);
        let query = self.search_query(false, req.samples(), &filter, req.threshold().is_some());

        let mut response = self
            .surreal
            .query(query.as_str())
            .bind(("vec", embedded_query))
            .bind(("tablename", self.documents_table.clone()))
            .bind(("threshold", req.threshold().unwrap_or(0.)))
            .await
            .map_err(|e| VectorStoreError::DatastoreError(Box::new(e)))?;

        let rows: Vec<SearchResultOnlyId> = response
            .take(0)
            .map_err(|e| VectorStoreError::DatastoreError(Box::new(e)))?;

        let rows: Vec<(f64, String)> = rows
            .into_iter()
            .map(|row| (row.distance, row.id.to_sql()))
            .collect();

        Ok(rows)
    }
}

#[cfg(test)]
mod tests {
    use super::{Mem, SurrealDistanceFunction, SurrealVectorStore};
    use rig::embeddings::{Embedding, EmbeddingError, EmbeddingModel};
    use surrealdb::Surreal;
    use surrealdb::engine::local::Db;

    #[derive(Clone, Copy)]
    struct TestEmbeddingModel;

    impl EmbeddingModel for TestEmbeddingModel {
        const MAX_DOCUMENTS: usize = 1;

        type Client = ();

        fn make(_: &Self::Client, _: impl Into<String>, _: Option<usize>) -> Self {
            Self
        }

        fn ndims(&self) -> usize {
            3
        }

        async fn embed_texts(
            &self,
            texts: impl IntoIterator<Item = String> + rig::wasm_compat::WasmCompatSend,
        ) -> Result<Vec<Embedding>, EmbeddingError> {
            Ok(texts
                .into_iter()
                .map(|text| Embedding {
                    document: text,
                    vec: vec![0.0, 0.0, 0.0],
                })
                .collect())
        }
    }

    async fn vector_store(
        distance_function: SurrealDistanceFunction,
    ) -> SurrealVectorStore<Db, TestEmbeddingModel> {
        let surreal = Surreal::new::<Mem>(()).await.unwrap();
        SurrealVectorStore::new(TestEmbeddingModel, surreal, None, distance_function)
    }

    #[test]
    fn index_backed_distances_share_the_knn_query_shape() {
        for distance_function in [
            SurrealDistanceFunction::Knn,
            SurrealDistanceFunction::Cosine,
            SurrealDistanceFunction::Euclidean,
            SurrealDistanceFunction::Hamming,
        ] {
            assert!(distance_function.uses_index_distance());
            assert_eq!(distance_function.knn_clause(4), "embedding <|4,4|> $vec");
            assert_eq!(
                distance_function.select_distance_expression(),
                "vector::distance::knn()"
            );
            assert_eq!(distance_function.threshold_operator(), "<=");
            assert_eq!(distance_function.sort_direction(), "ASC");
        }
    }

    #[test]
    fn jaccard_queries_keep_similarity_semantics() {
        assert!(!SurrealDistanceFunction::Jaccard.uses_index_distance());
        assert_eq!(SurrealDistanceFunction::Jaccard.sort_direction(), "DESC");
        assert_eq!(SurrealDistanceFunction::Jaccard.threshold_operator(), ">=");
        assert_eq!(SurrealDistanceFunction::Jaccard.knn_clause(4), "true");
        assert_eq!(
            SurrealDistanceFunction::Jaccard.select_distance_expression(),
            "vector::similarity::jaccard($vec, embedding)"
        );
    }

    #[tokio::test]
    async fn cosine_threshold_queries_wrap_index_backed_inner_query() {
        let store = vector_store(SurrealDistanceFunction::Cosine).await;

        assert_eq!(
            store.search_query(true, 4, "flag = true", true),
            "SELECT id, document, distance FROM (SELECT id, document, vector::distance::knn() AS distance FROM type::table($tablename) WHERE flag = true AND embedding <|4,4|> $vec ORDER BY distance ASC LIMIT 4) WHERE distance <= $threshold ORDER BY distance ASC LIMIT 4"
        );
    }

    #[tokio::test]
    async fn unfiltered_queries_do_not_prefix_knn_with_true() {
        let store = vector_store(SurrealDistanceFunction::Cosine).await;

        assert_eq!(
            store.search_query(true, 4, "true", false),
            "SELECT id, document, vector::distance::knn() AS distance FROM type::table($tablename) WHERE embedding <|4,4|> $vec ORDER BY distance ASC LIMIT 4"
        );
    }

    #[test]
    fn migration_example_tracks_surrealdb_3_requirements() {
        let migration = include_str!("../examples/migrations.surql");

        assert!(migration.contains("HNSW DIMENSION 1536"));
        assert!(!migration.contains("MTREE DIMENSION 1536"));
        assert!(migration.contains("SurrealDistanceFunction"));
        assert!(migration.contains("DIST must match"));
    }
}
