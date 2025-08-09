import axios, { AxiosInstance, AxiosRequestConfig, AxiosResponse } from 'axios';
import { toast } from 'react-hot-toast';
import {
  GenerationRequest,
  GenerationResponse,
  ValidationRequest,
  ValidationResult,
  Job,
  GeneratedDataset,
  MetricsData,
  SystemMetrics,
  ApiResponse,
  PaginatedResponse,
  FilterOptions,
  ExportOptions,
} from '../types';

// API 클라이언트 설정
class ApiClient {
  private client: AxiosInstance;

  constructor() {
    this.client = axios.create({
      baseURL: import.meta.env.VITE_API_URL || '/api/v1',
      timeout: 30000,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    // 요청 인터셉터
    this.client.interceptors.request.use(
      (config) => {
        const token = localStorage.getItem('access_token');
        if (token) {
          config.headers.Authorization = `Bearer ${token}`;
        }
        return config;
      },
      (error) => {
        return Promise.reject(error);
      }
    );

    // 응답 인터셉터
    this.client.interceptors.response.use(
      (response) => response,
      (error) => {
        if (error.response) {
          const { status, data } = error.response;

          if (status === 401) {
            localStorage.removeItem('access_token');
            window.location.href = '/login';
            toast.error('인증이 만료되었습니다. 다시 로그인해주세요.');
          } else if (status === 429) {
            toast.error('요청이 너무 많습니다. 잠시 후 다시 시도해주세요.');
          } else if (status >= 500) {
            toast.error('서버 오류가 발생했습니다.');
          } else if (data?.message) {
            toast.error(data.message);
          }
        } else if (error.request) {
          toast.error('서버에 연결할 수 없습니다.');
        }

        return Promise.reject(error);
      }
    );
  }

  // Generation API
  async generateData(request: GenerationRequest): Promise<GenerationResponse> {
    const response = await this.client.post<GenerationResponse>(
      '/generation/generate',
      request
    );
    return response.data;
  }

  async getJobStatus(jobId: string): Promise<Job> {
    const response = await this.client.get<Job>(`/generation/status/${jobId}`);
    return response.data;
  }

  async getGenerationResult(
    jobId: string,
    limit: number = 100,
    offset: number = 0
  ): Promise<GeneratedDataset> {
    const response = await this.client.get<GeneratedDataset>(
      `/generation/result/${jobId}`,
      {
        params: { limit, offset },
      }
    );
    return response.data;
  }

  async cancelGeneration(jobId: string): Promise<void> {
    await this.client.post(`/generation/cancel/${jobId}`);
  }

  async listJobs(
    filters?: FilterOptions,
    page: number = 1,
    pageSize: number = 50
  ): Promise<PaginatedResponse<Job>> {
    const params: any = {
      limit: pageSize,
      offset: (page - 1) * pageSize,
    };

    if (filters?.status) {
      params.status = filters.status;
    }

    if (filters?.search) {
      params.search = filters.search;
    }

    const response = await this.client.get<PaginatedResponse<Job>>(
      '/generation/jobs',
      { params }
    );
    return response.data;
  }

  async regenerateData(
    jobId: string,
    modifications?: Partial<GenerationRequest>
  ): Promise<{ new_job_id: string }> {
    const response = await this.client.post<{ new_job_id: string }>(
      `/generation/regenerate/${jobId}`,
      modifications
    );
    return response.data;
  }

  // Validation API
  async validateData(request: ValidationRequest): Promise<ValidationResult> {
    const response = await this.client.post<ValidationResult>(
      '/validation/validate',
      request
    );
    return response.data;
  }

  async getValidationHistory(
    jobId: string,
    limit: number = 10
  ): Promise<ValidationResult[]> {
    const response = await this.client.get<{ history: ValidationResult[] }>(
      `/validation/validate/${jobId}/history`,
      {
        params: { limit },
      }
    );
    return response.data.history;
  }

  async batchValidate(
    jobIds: string[],
    validationType: string = 'all'
  ): Promise<any> {
    const response = await this.client.post('/validation/validate/batch', {
      job_ids: jobIds,
      validation_type: validationType,
    });
    return response.data;
  }

  async getQualityThresholds(): Promise<any> {
    const response = await this.client.get('/validation/quality-thresholds');
    return response.data;
  }

  // Metrics API
  async collectMetrics(
    jobIds?: string[],
    metricTypes: string[] = ['quality', 'bias', 'performance'],
    startDate?: Date,
    endDate?: Date
  ): Promise<MetricsData> {
    const response = await this.client.post<MetricsData>('/metrics/collect', {
      job_ids: jobIds,
      metric_types: metricTypes,
      start_date: startDate?.toISOString(),
      end_date: endDate?.toISOString(),
    });
    return response.data;
  }

  async getMetricsSummary(period: string = '7d'): Promise<any> {
    const response = await this.client.get('/metrics/summary', {
      params: { period },
    });
    return response.data;
  }

  async getRealtimeMetrics(): Promise<SystemMetrics> {
    const response = await this.client.get<SystemMetrics>('/metrics/realtime');
    return response.data;
  }

  async getMetricTrends(
    metricType: string,
    granularity: string = 'hour',
    period: number = 24
  ): Promise<any> {
    const response = await this.client.get(`/metrics/trends/${metricType}`, {
      params: { granularity, period },
    });
    return response.data;
  }

  async compareMetrics(jobIds: string[]): Promise<any> {
    const response = await this.client.get('/metrics/comparison', {
      params: { job_ids: jobIds },
    });
    return response.data;
  }

  async getBenchmarks(domain?: string): Promise<any> {
    const response = await this.client.get('/metrics/benchmarks', {
      params: domain ? { domain } : undefined,
    });
    return response.data;
  }

  async exportMetrics(options: ExportOptions): Promise<Blob> {
    const response = await this.client.get('/metrics/export', {
      params: {
        format: options.format,
        start_date: options.dateRange?.start.toISOString(),
        end_date: options.dateRange?.end.toISOString(),
      },
      responseType: 'blob',
    });
    return response.data;
  }

  // Health Check
  async checkHealth(): Promise<any> {
    const response = await this.client.get('/health');
    return response.data;
  }

  // System Info
  async getSystemInfo(): Promise<any> {
    const response = await this.client.get('/info');
    return response.data;
  }

  // File Download
  async downloadFile(url: string): Promise<Blob> {
    const response = await this.client.get(url, {
      responseType: 'blob',
    });
    return response.data;
  }

  // WebSocket Connection
  connectWebSocket(onMessage: (message: any) => void): WebSocket {
    const wsUrl = import.meta.env.VITE_WS_URL || 'ws://localhost:8000/ws';
    const ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      console.log('WebSocket connected');
    };

    ws.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data);
        onMessage(message);
      } catch (error) {
        console.error('Failed to parse WebSocket message:', error);
      }
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      toast.error('실시간 연결에 문제가 발생했습니다.');
    };

    ws.onclose = () => {
      console.log('WebSocket disconnected');
      // 자동 재연결 로직
      setTimeout(() => {
        this.connectWebSocket(onMessage);
      }, 5000);
    };

    return ws;
  }
}

// 싱글톤 인스턴스
const apiClient = new ApiClient();

export default apiClient;

// 편의 함수들
export const api = {
  generation: {
    create: apiClient.generateData.bind(apiClient),
    getStatus: apiClient.getJobStatus.bind(apiClient),
    getResult: apiClient.getGenerationResult.bind(apiClient),
    cancel: apiClient.cancelGeneration.bind(apiClient),
    list: apiClient.listJobs.bind(apiClient),
    regenerate: apiClient.regenerateData.bind(apiClient),
  },
  validation: {
    validate: apiClient.validateData.bind(apiClient),
    getHistory: apiClient.getValidationHistory.bind(apiClient),
    batchValidate: apiClient.batchValidate.bind(apiClient),
    getThresholds: apiClient.getQualityThresholds.bind(apiClient),
  },
  metrics: {
    collect: apiClient.collectMetrics.bind(apiClient),
    getSummary: apiClient.getMetricsSummary.bind(apiClient),
    getRealtime: apiClient.getRealtimeMetrics.bind(apiClient),
    getTrends: apiClient.getMetricTrends.bind(apiClient),
    compare: apiClient.compareMetrics.bind(apiClient),
    getBenchmarks: apiClient.getBenchmarks.bind(apiClient),
    export: apiClient.exportMetrics.bind(apiClient),
  },
  system: {
    checkHealth: apiClient.checkHealth.bind(apiClient),
    getInfo: apiClient.getSystemInfo.bind(apiClient),
  },
  ws: {
    connect: apiClient.connectWebSocket.bind(apiClient),
  },
};