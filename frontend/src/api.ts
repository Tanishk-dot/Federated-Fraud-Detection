const BASE = "/api"

export interface Transaction {
  amount: number
  hour: number
  day_of_week: number
  merchant_type: string
  distance_km: number
  minutes_since_last: number
  card_age_days: number
  txns_last_24h: number
}

export interface PredictResponse {
  fraud_prob: number
  anomaly_score: number
  decision: "FRAUD" | "LEGITIMATE"
}

export interface ModelInfo {
  loaded: boolean
  checkpoint?: string
  final_metrics?: Record<string, number | number[][]>
  num_rounds?: number
  raw_columns: string[]
  merchant_types: string[]
  defaults: Record<string, number | string>
  message?: string
}

export interface DatasetInfo {
  name: string
  num_clients: number
}

export interface CsvInspectResponse {
  columns: string[]
  num_rows: number
  sample_rows: Record<string, unknown>[]
  suggested_mapping: Record<string, string | null>
  required_fields: string[]
  merchant_types: string[]
  defaults: Record<string, number | string>
}

export interface CsvPredictResult {
  row: number
  account_id?: string
  fraud_prob: number
  anomaly_score: number
  decision: "FRAUD" | "LEGITIMATE"
}

export interface CsvPredictResponse {
  results: CsvPredictResult[]
  num_rows: number
  total_rows_in_file: number
  truncated: boolean
  truncation_note: string | null
  num_fraud: number
  mapping_used: Record<string, string | null>
  unmapped_fields: string[]
}

export interface TrainRequest {
  dataset: string
  rounds: number
  clients_per_round: number
  hidden_dim: number
  max_samples: number
  dp_enabled: boolean
  epsilon: number
}

export interface TrainRoundResult {
  round: number
  selected_clients: number[]
  accuracy: number
  precision: number
  recall: number
  f1: number
  roc_auc: number
}

export interface TrainJobStatus {
  status: "starting" | "running" | "complete" | "error"
  round: number
  total_rounds: number
  history: TrainRoundResult[]
  final: TrainRoundResult | null
  error: string | null
}

export interface SimulationRequest {
  num_clients: number
  rounds: number
  clients_per_round: number
  samples_per_client: number
  fraud_rate: number
  hidden_dim: number
  dp_enabled: boolean
  epsilon: number
}

// All fields optional/union across every event "type" this simulation emits -
// see backend/main.py's _run_simulation_job for the exact shapes.
export interface SimEvent {
  type: string
  round?: number
  total_rounds?: number
  clients?: number | number[]
  client?: number
  attempt?: number
  samples_per_client?: number
  fraud_rate?: number
  checkpoint?: string | null
  epsilon?: number
  clip_norm?: number
  delta?: number
  accuracy?: number
  f1?: number
  recall?: number
  roc_auc?: number
  amount?: number
  merchant_type?: string
  hour?: number
  fraud_prob?: number
  raw_decision?: "FRAUD" | "LEGITIMATE"
  decision?: "FRAUD" | "LEGITIMATE"
  actual?: "FRAUD" | "LEGITIMATE"
  correct?: boolean
}

export interface SimJobStatus {
  status: "starting" | "running" | "complete" | "error"
  round: number
  total_rounds: number
  events: SimEvent[]
  final_metrics: SimEvent | null
  error: string | null
}

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`)
  if (!res.ok) throw new Error(`${path}: ${res.status} ${await res.text()}`)
  return res.json()
}

async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  })
  if (!res.ok) throw new Error(`${path}: ${res.status} ${await res.text()}`)
  return res.json()
}

export const api = {
  health: () => get<{ status: string }>("/health"),
  modelInfo: () => get<ModelInfo>("/model-info"),
  datasets: () => get<{ datasets: DatasetInfo[] }>("/datasets"),
  trainingHistory: (dataset: string) =>
    get<{ dataset: string; history: any }>(`/training-history/${dataset}`),
  predict: (txn: Transaction) => post<PredictResponse>("/predict", txn),

  csvInspect: async (file: File): Promise<CsvInspectResponse> => {
    const fd = new FormData()
    fd.append("file", file)
    const res = await fetch(`${BASE}/csv/inspect`, { method: "POST", body: fd })
    if (!res.ok) throw new Error(await res.text())
    return res.json()
  },

  csvPredict: async (
    file: File,
    mapping: Record<string, string | null>,
    accountIdCol?: string | null
  ): Promise<CsvPredictResponse> => {
    const fd = new FormData()
    fd.append("file", file)
    fd.append("mapping", JSON.stringify(mapping))
    if (accountIdCol) fd.append("account_id_col", accountIdCol)
    const res = await fetch(`${BASE}/csv/predict`, { method: "POST", body: fd })
    if (!res.ok) throw new Error(await res.text())
    return res.json()
  },

  trainStart: (req: TrainRequest) => post<{ job_id: string }>("/train/start", req),
  trainStatus: (jobId: string) => get<TrainJobStatus>(`/train/status/${jobId}`),

  simStart: (req: Partial<SimulationRequest> = {}) => post<{ job_id: string }>("/simulation/start", req),
  simStatus: (jobId: string) => get<SimJobStatus>(`/simulation/status/${jobId}`),
}
