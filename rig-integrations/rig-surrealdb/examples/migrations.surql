-- define table & fields
DEFINE TABLE documents SCHEMAFULL;
DEFINE field document on table documents type object FLEXIBLE;
DEFINE field embedding on table documents type array<float>;
DEFINE field embedded_text on table documents type string;

-- define index on embedding field
-- DIST must match the SurrealDistanceFunction used by rig-surrealdb.
-- For example, SurrealDistanceFunction::Cosine expects DIST COSINE.
DEFINE INDEX IF NOT EXISTS words_embedding_vector_index ON documents
     FIELDS embedding
     HNSW DIMENSION 1536
     DIST COSINE;
