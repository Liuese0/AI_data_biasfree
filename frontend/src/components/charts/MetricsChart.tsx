import React, { useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  Filler,
} from 'chart.js';
import { Line, Bar } from 'react-chartjs-2';
import { api } from '../../services/api';
import LoadingSpinner from '../common/LoadingSpinner';

// Chart.js 등록
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

interface MetricsChartProps {
  type: 'quality' | 'bias' | 'performance' | 'generation_volume';
  period: string;
  height?: number;
  showLegend?: boolean;
}

const MetricsChart: React.FC<MetricsChartProps> = ({
  type,
  period,
  height = 400,
  showLegend = true,
}) => {
  // 기간에 따른 granularity 설정
  const granularity = useMemo(() => {
    switch (period) {
      case '1d':
        return 'hour';
      case '7d':
        return 'day';
      case '30d':
        return 'day';
      case '90d':
        return 'week';
      default:
        return 'day';
    }
  }, [period]);

  const periodValue = useMemo(() => {
    switch (period) {
      case '1d':
        return 24;
      case '7d':
        return 7;
      case '30d':
        return 30;
      case '90d':
        return 13; // 13 weeks
      default:
        return 7;
    }
  }, [period]);

  // 트렌드 데이터 조회
  const { data, isLoading, error } = useQuery({
    queryKey: ['metric-trends', type, granularity, periodValue],
    queryFn: () => api.metrics.getTrends(type, granularity, periodValue),
  });

  // 차트 데이터 변환
  const chartData = useMemo(() => {
    if (!data?.data_points) {
      return null;
    }

    const labels = data.data_points.map((point: any) => {
      if (granularity === 'hour') {
        return new Date(point.period).toLocaleTimeString('ko-KR', {
          hour: '2-digit',
          minute: '2-digit',
        });
      } else if (granularity === 'day') {
        return new Date(point.period).toLocaleDateString('ko-KR', {
          month: 'short',
          day: 'numeric',
        });
      } else {
        return `Week ${point.period.split('-')[1]}`;
      }
    });

    const values = data.data_points.map((point: any) => point.value);

    // 차트 색상 설정
    const colors = {
      quality: {
        border: 'rgb(34, 197, 94)',
        background: 'rgba(34, 197, 94, 0.1)',
      },
      bias: {
        border: 'rgb(239, 68, 68)',
        background: 'rgba(239, 68, 68, 0.1)',
      },
      performance: {
        border: 'rgb(59, 130, 246)',
        background: 'rgba(59, 130, 246, 0.1)',
      },
      generation_volume: {
        border: 'rgb(168, 85, 247)',
        background: 'rgba(168, 85, 247, 0.1)',
      },
    };

    const color = colors[type];

    return {
      labels,
      datasets: [
        {
          label: getMetricLabel(type),
          data: values,
          borderColor: color.border,
          backgroundColor: color.background,
          borderWidth: 2,
          fill: true,
          tension: 0.4,
          pointRadius: 3,
          pointHoverRadius: 5,
        },
      ],
    };
  }, [data, type, granularity]);

  // 차트 옵션
  const options = useMemo(
    () => ({
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          display: showLegend,
          position: 'top' as const,
        },
        tooltip: {
          mode: 'index' as const,
          intersect: false,
          callbacks: {
            label: (context: any) => {
              let label = context.dataset.label || '';
              if (label) {
                label += ': ';
              }
              if (type === 'quality' || type === 'bias') {
                label += (context.parsed.y * 100).toFixed(1) + '%';
              } else if (type === 'generation_volume') {
                label += context.parsed.y.toLocaleString() + ' 샘플';
              } else {
                label += context.parsed.y.toFixed(2);
              }
              return label;
            },
          },
        },
      },
      scales: {
        x: {
          display: true,
          grid: {
            display: false,
          },
        },
        y: {
          display: true,
          grid: {
            color: 'rgba(0, 0, 0, 0.05)',
          },
          ticks: {
            callback: function (value: any) {
              if (type === 'quality' || type === 'bias') {
                return (value * 100).toFixed(0) + '%';
              } else if (type === 'generation_volume') {
                return value.toLocaleString();
              }
              return value;
            },
          },
        },
      },
    }),
    [type, showLegend]
  );

  if (isLoading) {
    return (
      <div className="flex items-center justify-center" style={{ height }}>
        <LoadingSpinner size="medium" />
      </div>
    );
  }

  if (error || !chartData) {
    return (
      <div className="flex items-center justify-center" style={{ height }}>
        <p className="text-gray-500">차트 데이터를 불러올 수 없습니다</p>
      </div>
    );
  }

  const ChartComponent = type === 'generation_volume' ? Bar : Line;

  return (
    <div style={{ height }}>
      <ChartComponent data={chartData} options={options} />
    </div>
  );
};

// 메트릭 라벨 반환
function getMetricLabel(type: string): string {
  switch (type) {
    case 'quality':
      return '품질 점수';
    case 'bias':
      return '편향성 점수';
    case 'performance':
      return '처리 성능';
    case 'generation_volume':
      return '생성량';
    default:
      return type;
  }
}

export default MetricsChart;