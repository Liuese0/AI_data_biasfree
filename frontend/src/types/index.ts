// API Types
export interface GenerationRequest {
  prompt: string;
  data_type: DataType;
  domain: Domain;
  quantity: number;
  output_format: OutputFormat;
  bias_mitigation_config: BiasMitigationConfig;
  quality_constraints: QualityConstraints;
  domain_constraints?: Record<string, any>;
  ensemble_config: EnsembleConfig;
}

export interface GenerationResponse {
  job_id: string;
  status: JobStatus;
  created_at: string;
  estimated_completion?: string;
  progress: number;
  message: string;
}

export interface Job {
  job_id: string;
  prompt: string;
  data_type: DataType;
  domain: Domain;
  quantity: number;
  status: JobStatus;
  progress: number;
  created_at: string;
  completed_at?: string;
  download_url?: string;
}

export interface ValidationResult {
  job_id: string;
  validation_timestamp: string;
  quality_metrics: Record<string, number>;
  bias_metrics: Record<string, any>;
  privacy_metrics: Record<string, any>;
  overall_quality_score: number;
  overall_bias_score: number;
  privacy_compliance: boolean;
  recommendations: string[];
  warnings: string[];
}

export interface MetricsData {
  timestamp: string;
  metrics: {
    quality?: QualityMetrics;
    bias?: BiasMetrics;
    performance?: PerformanceMetrics;
  };
  aggregated_stats: Record<string, number>;
  trends: Record<string, number[]>;
}

export interface QualityMetrics {
  count: number;
  average: number;
  median: number;
  std: number;
  min: number;
  max: number;
  percentiles: Record<string, number>;
}

export interface BiasMetrics {
  count: number;
  average: number;
  median: number;
  std: number;
  min: number;
  max: number;
  category_averages: Record<string, number>;
  high_bias_count: number;
}

export interface PerformanceMetrics {
  count: number;
  total_samples_generated: number;
  average_processing_time: number;
  median_processing_time: number;
  average_generation_rate: number;
  success_rate: number;
}

export interface DataSample {
  sample_id: string;
  content: any;
  metadata: Record<string, any>;
  quality_score: number;
  bias_indicators: Record<string, number>;
}

export interface GeneratedDataset {
  job_id: string;
  status: JobStatus;
  data_type: DataType;
  domain: Domain;
  quantity: number;
  actual_quantity: number;
  samples: DataSample[];
  generation_metadata: Record<string, any>;
  validation_result?: ValidationResult;
  statistics: Record<string, any>;
  created_at: string;
  completed_at?: string;
  download_url?: string;
}

// Enums
export enum DataType {
  TEXT = 'text',
  IMAGE = 'image',
  TABULAR = 'tabular',
}

export enum Domain {
  MEDICAL = 'medical',
  FINANCIAL = 'financial',
  LEGAL = 'legal',
  GENERAL = 'general',
}

export enum OutputFormat {
  JSON = 'json',
  CSV = 'csv',
  PARQUET = 'parquet',
  TXT = 'txt',
}

export enum JobStatus {
  PENDING = 'pending',
  PROCESSING = 'processing',
  COMPLETED = 'completed',
  FAILED = 'failed',
  VALIDATED = 'validated',
  CANCELLED = 'cancelled',
}

// Configuration Types
export interface BiasMitigationConfig {
  demographic_balance: boolean;
  gender_balance: boolean;
  age_balance: boolean;
  cultural_diversity: boolean;
  target_bias_reduction: number;
}

export interface QualityConstraints {
  min_quality_score: number;
  min_semantic_similarity: number;
  max_reidentification_risk: number;
}

export interface EnsembleConfig {
  num_generators: number;
  voting_method: string;
  diversity_weight: number;
}

// UI Types
export interface ChartData {
  labels: string[];
  datasets: {
    label: string;
    data: number[];
    borderColor?: string;
    backgroundColor?: string;
    tension?: number;
  }[];
}

export interface TableColumn<T = any> {
  key: string;
  label: string;
  sortable?: boolean;
  render?: (value: any, row: T) => React.ReactNode;
}

export interface PaginationInfo {
  page: number;
  pageSize: number;
  total: number;
  totalPages: number;
}

export interface FilterOptions {
  search?: string;
  status?: JobStatus;
  domain?: Domain;
  dateRange?: {
    start: Date;
    end: Date;
  };
}

export interface NotificationOptions {
  type: 'success' | 'error' | 'warning' | 'info';
  title: string;
  message?: string;
  duration?: number;
}

// Store Types
export interface AppState {
  isInitialized: boolean;
  isLoading: boolean;
  error: string | null;
  user: User | null;
  notifications: Notification[];
  settings: AppSettings;
}

export interface User {
  id: string;
  email: string;
  name: string;
  roles: string[];
}

export interface Notification {
  id: string;
  type: 'success' | 'error' | 'warning' | 'info';
  title: string;
  message?: string;
  timestamp: Date;
  read: boolean;
}

export interface AppSettings {
  theme: 'light' | 'dark' | 'auto';
  language: string;
  notifications: {
    email: boolean;
    push: boolean;
    inApp: boolean;
  };
  defaults: {
    domain: Domain;
    dataType: DataType;
    outputFormat: OutputFormat;
    quantity: number;
  };
}

// Form Types
export interface GenerationFormData {
  prompt: string;
  dataType: DataType;
  domain: Domain;
  quantity: number;
  outputFormat: OutputFormat;
  enableBiasMitigation: boolean;
  biasMitigation: {
    demographicBalance: boolean;
    genderBalance: boolean;
    ageBalance: boolean;
    culturalDiversity: boolean;
    targetBiasReduction: number;
  };
  qualityConstraints: {
    minQualityScore: number;
    minSemanticSimilarity: number;
    maxReidentificationRisk: number;
  };
  ensemble: {
    numGenerators: number;
    votingMethod: string;
    diversityWeight: number;
  };
  advancedOptions?: {
    temperature?: number;
    maxLength?: number;
    topP?: number;
  };
}

// API Response Types
export interface ApiResponse<T = any> {
  data?: T;
  error?: string;
  message?: string;
  status: number;
}

export interface ApiError {
  error_code: string;
  message: string;
  detail?: string;
  timestamp: string;
}

export interface PaginatedResponse<T = any> {
  total: number;
  limit: number;
  offset: number;
  data: T[];
}

// Metrics Types
export interface SystemMetrics {
  system_status: {
    ensemble_ready: boolean;
    active_generators: number;
    total_generations: number;
  };
  active_jobs: {
    count: number;
    jobs: Array<{
      job_id: string;
      status: string;
      progress: number;
      current_step: string;
    }>;
  };
  current_performance: {
    average_quality_score: number;
    average_bias_score: number;
  };
}

export interface Benchmark {
  domain: string;
  benchmark: {
    target_quality: number;
    max_bias: number;
    privacy_compliance: number;
    generation_speed: number;
  };
  current_performance?: {
    quality: number;
    bias: number;
    privacy_compliance: number;
    generation_speed: number;
  };
  comparison?: Record<string, string>;
}

export interface TrendData {
  metric_type: string;
  granularity: string;
  period: number;
  start_date: string;
  end_date: string;
  data_points: Array<{
    period: string;
    value: number;
  }>;
}

// WebSocket Types
export interface WebSocketMessage {
  type: 'job_update' | 'metric_update' | 'notification';
  payload: any;
  timestamp: string;
}

// Export Types
export interface ExportOptions {
  format: 'json' | 'csv' | 'excel';
  includeMetadata: boolean;
  dateRange?: {
    start: Date;
    end: Date;
  };
  fields?: string[];
}