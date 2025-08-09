import React, { useEffect, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Link } from 'react-router-dom';
import {
  ChartBarIcon,
  CubeIcon,
  CheckCircleIcon,
  ClockIcon,
  ExclamationTriangleIcon,
  ArrowUpIcon,
  ArrowDownIcon,
} from '@heroicons/react/24/outline';
import { api } from '../services/api';
import LoadingSpinner from '../components/common/LoadingSpinner';
import MetricsChart from '../components/charts/MetricsChart';
import JobsTable from '../components/tables/JobsTable';
import { SystemMetrics, Job } from '../types';
import { formatDistanceToNow, formatDateTime } from '../utils/date';

const Dashboard: React.FC = () => {
  const [period, setPeriod] = useState('7d');

  // 실시간 메트릭 조회
  const { data: realtimeMetrics, isLoading: isLoadingRealtime } = useQuery<SystemMetrics>({
    queryKey: ['realtime-metrics'],
    queryFn: api.metrics.getRealtime,
    refetchInterval: 5000, // 5초마다 업데이트
  });

  // 메트릭 요약 조회
  const { data: metricsSummary, isLoading: isLoadingS ummary } = useQuery({
    queryKey: ['metrics-summary', period],
    queryFn: () => api.metrics.getSummary(period),
  });

  // 최근 작업 조회
  const { data: recentJobs, isLoading: isLoadingJobs } = useQuery({
    queryKey: ['recent-jobs'],
    queryFn: () => api.generation.list(undefined, 1, 5),
  });

  // 시스템 상태 조회
  const { data: systemHealth } = useQuery({
    queryKey: ['system-health'],
    queryFn: api.system.checkHealth,
    refetchInterval: 30000, // 30초마다 체크
  });

  const statsCards = [
    {
      title: '활성 생성기',
      value: realtimeMetrics?.system_status.active_generators || 0,
      icon: CubeIcon,
      color: 'blue',
      change: '+2',
      changeType: 'increase' as const,
    },
    {
      title: '총 생성 데이터',
      value: metricsSummary?.job_statistics?.total_samples_generated || 0,
      icon: ChartBarIcon,
      color: 'green',
      change: '+12.5%',
      changeType: 'increase' as const,
    },
    {
      title: '평균 품질 점수',
      value: `${((realtimeMetrics?.current_performance.average_quality_score || 0) * 100).toFixed(1)}%`,
      icon: CheckCircleIcon,
      color: 'purple',
      change: '+3.2%',
      changeType: 'increase' as const,
    },
    {
      title: '평균 편향성',
      value: `${((realtimeMetrics?.current_performance.average_bias_score || 0) * 100).toFixed(1)}%`,
      icon: ExclamationTriangleIcon,
      color: 'yellow',
      change: '-5.1%',
      changeType: 'decrease' as const,
    },
  ];

  const isLoading = isLoadingRealtime || isLoadingSummary || isLoadingJobs;

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-96">
        <LoadingSpinner size="large" message="대시보드 로딩 중..." />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">대시보드</h1>
          <p className="mt-1 text-sm text-gray-500">
            시스템 상태 및 주요 메트릭 개요
          </p>
        </div>
        
        {/* Period Selector */}
        <div className="flex items-center space-x-2">
          <label className="text-sm text-gray-700">기간:</label>
          <select
            value={period}
            onChange={(e) => setPeriod(e.target.value)}
            className="input py-1 px-2 text-sm"
          >
            <option value="1d">1일</option>
            <option value="7d">7일</option>
            <option value="30d">30일</option>
            <option value="90d">90일</option>
          </select>
        </div>
      </div>

      {/* System Status Alert */}
      {systemHealth?.status !== 'healthy' && (
        <div className="bg-yellow-50 border-l-4 border-yellow-400 p-4">
          <div className="flex">
            <ExclamationTriangleIcon className="h-5 w-5 text-yellow-400" />
            <div className="ml-3">
              <p className="text-sm text-yellow-700">
                시스템 상태: {systemHealth?.status || 'unknown'}
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Stats Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {statsCards.map((stat, index) => (
          <div key={index} className="card">
            <div className="card-body">
              <div className="flex items-center justify-between">
                <div className="flex-1">
                  <p className="text-sm font-medium text-gray-600">{stat.title}</p>
                  <p className="mt-2 text-3xl font-semibold text-gray-900">
                    {stat.value}
                  </p>
                  <div className="mt-2 flex items-center text-sm">
                    {stat.changeType === 'increase' ? (
                      <ArrowUpIcon className="h-4 w-4 text-green-500" />
                    ) : (
                      <ArrowDownIcon className="h-4 w-4 text-red-500" />
                    )}
                    <span
                      className={`ml-1 ${
                        stat.changeType === 'increase'
                          ? 'text-green-600'
                          : 'text-red-600'
                      }`}
                    >
                      {stat.change}
                    </span>
                    <span className="ml-2 text-gray-500">지난 기간 대비</span>
                  </div>
                </div>
                <div className="flex-shrink-0">
                  <div
                    className={`p-3 rounded-full bg-${stat.color}-100 text-${stat.color}-600`}
                  >
                    <stat.icon className="h-6 w-6" />
                  </div>
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Charts Section */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Quality Trend Chart */}
        <div className="card">
          <div className="card-body">
            <h3 className="text-lg font-medium text-gray-900 mb-4">
              품질 점수 추이
            </h3>
            <MetricsChart
              type="quality"
              period={period}
              height={300}
            />
          </div>
        </div>

        {/* Bias Trend Chart */}
        <div className="card">
          <div className="card-body">
            <h3 className="text-lg font-medium text-gray-900 mb-4">
              편향성 점수 추이
            </h3>
            <MetricsChart
              type="bias"
              period={period}
              height={300}
            />
          </div>
        </div>
      </div>

      {/* Active Jobs */}
      {realtimeMetrics?.active_jobs.count > 0 && (
        <div className="card">
          <div className="card-body">
            <h3 className="text-lg font-medium text-gray-900 mb-4">
              진행 중인 작업
            </h3>
            <div className="space-y-3">
              {realtimeMetrics.active_jobs.jobs.map((job) => (
                <div
                  key={job.job_id}
                  className="flex items-center justify-between p-3 bg-gray-50 rounded-lg"
                >
                  <div className="flex items-center space-x-3">
                    <ClockIcon className="h-5 w-5 text-gray-400" />
                    <div>
                      <p className="text-sm font-medium text-gray-900">
                        작업 ID: {job.job_id.substring(0, 8)}...
                      </p>
                      <p className="text-xs text-gray-500">
                        단계: {job.current_step}
                      </p>
                    </div>
                  </div>
                  <div className="flex items-center space-x-4">
                    <div className="w-32">
                      <div className="progress-bar h-2">
                        <div
                          className="progress-bar-fill"
                          style={{ width: `${job.progress * 100}%` }}
                        />
                      </div>
                    </div>
                    <span className="text-sm text-gray-600">
                      {(job.progress * 100).toFixed(0)}%
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Recent Jobs Table */}
      <div className="card">
        <div className="card-body">
          <div className="flex justify-between items-center mb-4">
            <h3 className="text-lg font-medium text-gray-900">최근 작업</h3>
            <Link
              to="/jobs"
              className="text-sm text-primary-600 hover:text-primary-700"
            >
              모두 보기 →
            </Link>
          </div>
          {recentJobs?.data && recentJobs.data.length > 0 ? (
            <JobsTable jobs={recentJobs.data} compact />
          ) : (
            <p className="text-center text-gray-500 py-8">
              최근 작업이 없습니다
            </p>
          )}
        </div>
      </div>

      {/* System Stats */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Domain Distribution */}
        <div className="card">
          <div className="card-body">
            <h3 className="text-lg font-medium text-gray-900 mb-4">
              도메인별 분포
            </h3>
            {metricsSummary?.job_statistics?.domain_distribution && (
              <div className="space-y-2">
                {Object.entries(metricsSummary.job_statistics.domain_distribution).map(
                  ([domain, count]) => (
                    <div key={domain} className="flex justify-between">
                      <span className="text-sm text-gray-600 capitalize">
                        {domain}
                      </span>
                      <span className="text-sm font-medium text-gray-900">
                        {count as number}
                      </span>
                    </div>
                  )
                )}
              </div>
            )}
          </div>
        </div>

        {/* Quality Stats */}
        <div className="card">
          <div className="card-body">
            <h3 className="text-lg font-medium text-gray-900 mb-4">
              품질 통계
            </h3>
            <div className="space-y-2">
              <div className="flex justify-between">
                <span className="text-sm text-gray-600">평균 품질</span>
                <span className="text-sm font-medium text-gray-900">
                  {((metricsSummary?.quality_statistics?.average_quality || 0) * 100).toFixed(1)}%
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-sm text-gray-600">합격률</span>
                <span className="text-sm font-medium text-gray-900">
                  {((metricsSummary?.quality_statistics?.quality_pass_rate || 0) * 100).toFixed(1)}%
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-sm text-gray-600">검증 수</span>
                <span className="text-sm font-medium text-gray-900">
                  {metricsSummary?.quality_statistics?.total_validations || 0}
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* Performance Stats */}
        <div className="card">
          <div className="card-body">
            <h3 className="text-lg font-medium text-gray-900 mb-4">
              성능 통계
            </h3>
            <div className="space-y-2">
              <div className="flex justify-between">
                <span className="text-sm text-gray-600">생성 속도</span>
                <span className="text-sm font-medium text-gray-900">
                  {metricsSummary?.performance_statistics?.average_generation_rate?.toFixed(1) || 0} /분
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-sm text-gray-600">성공률</span>
                <span className="text-sm font-medium text-gray-900">
                  {((metricsSummary?.performance_statistics?.success_rate || 0) * 100).toFixed(1)}%
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-sm text-gray-600">총 작업</span>
                <span className="text-sm font-medium text-gray-900">
                  {metricsSummary?.job_statistics?.total_jobs || 0}
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;