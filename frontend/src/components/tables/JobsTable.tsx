import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import {
  ChevronUpIcon,
  ChevronDownIcon,
  EyeIcon,
  ArrowPathIcon,
  XMarkIcon,
} from '@heroicons/react/24/outline';
import { Job, JobStatus } from '../../types';
import { formatDateTime, formatDuration } from '../../utils/date';
import JobStatusBadge from '../common/JobStatusBadge';

interface JobsTableProps {
  jobs: Job[];
  compact?: boolean;
  onRefresh?: () => void;
  onCancel?: (jobId: string) => void;
  onRegenerate?: (jobId: string) => void;
}

const JobsTable: React.FC<JobsTableProps> = ({
  jobs,
  compact = false,
  onRefresh,
  onCancel,
  onRegenerate,
}) => {
  const [sortField, setSortField] = useState<string>('created_at');
  const [sortDirection, setSortDirection] = useState<'asc' | 'desc'>('desc');

  // 정렬 처리
  const sortedJobs = [...jobs].sort((a, b) => {
    const aValue = a[sortField as keyof Job];
    const bValue = b[sortField as keyof Job];

    if (aValue === null || aValue === undefined) return 1;
    if (bValue === null || bValue === undefined) return -1;

    if (aValue < bValue) return sortDirection === 'asc' ? -1 : 1;
    if (aValue > bValue) return sortDirection === 'asc' ? 1 : -1;
    return 0;
  });

  const handleSort = (field: string) => {
    if (sortField === field) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc');
    } else {
      setSortField(field);
      setSortDirection('asc');
    }
  };

  const SortIcon = ({ field }: { field: string }) => {
    if (sortField !== field) {
      return <div className="w-4 h-4" />;
    }
    return sortDirection === 'asc' ? (
      <ChevronUpIcon className="w-4 h-4" />
    ) : (
      <ChevronDownIcon className="w-4 h-4" />
    );
  };

  const columns = [
    { key: 'job_id', label: 'ID', sortable: true },
    { key: 'prompt', label: '프롬프트', sortable: false },
    { key: 'domain', label: '도메인', sortable: true },
    { key: 'data_type', label: '타입', sortable: true },
    { key: 'quantity', label: '수량', sortable: true },
    { key: 'status', label: '상태', sortable: true },
    { key: 'progress', label: '진행률', sortable: true },
    { key: 'created_at', label: '생성일시', sortable: true },
    { key: 'actions', label: '작업', sortable: false },
  ];

  const displayColumns = compact
    ? columns.filter((col) =>
        ['job_id', 'prompt', 'status', 'progress', 'actions'].includes(col.key)
      )
    : columns;

  return (
    <div className="overflow-x-auto">
      <table className="min-w-full divide-y divide-gray-200">
        <thead className="bg-gray-50">
          <tr>
            {displayColumns.map((column) => (
              <th
                key={column.key}
                className={`px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider ${
                  column.sortable ? 'cursor-pointer hover:bg-gray-100' : ''
                }`}
                onClick={() => column.sortable && handleSort(column.key)}
              >
                <div className="flex items-center space-x-1">
                  <span>{column.label}</span>
                  {column.sortable && <SortIcon field={column.key} />}
                </div>
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="bg-white divide-y divide-gray-200">
          {sortedJobs.map((job) => (
            <tr key={job.job_id} className="hover:bg-gray-50">
              {displayColumns.includes(columns[0]) && (
                <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                  <Link
                    to={`/jobs/${job.job_id}`}
                    className="text-primary-600 hover:text-primary-700"
                  >
                    {job.job_id.substring(0, 8)}...
                  </Link>
                </td>
              )}
              
              {displayColumns.includes(columns[1]) && (
                <td className="px-6 py-4 text-sm text-gray-900">
                  <div className="max-w-xs truncate" title={job.prompt}>
                    {job.prompt}
                  </div>
                </td>
              )}
              
              {displayColumns.includes(columns[2]) && (
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                  <span className="capitalize">{job.domain}</span>
                </td>
              )}
              
              {displayColumns.includes(columns[3]) && (
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                  <span className="capitalize">{job.data_type}</span>
                </td>
              )}
              
              {displayColumns.includes(columns[4]) && (
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                  {job.quantity.toLocaleString()}
                </td>
              )}
              
              {displayColumns.includes(columns[5]) && (
                <td className="px-6 py-4 whitespace-nowrap">
                  <JobStatusBadge status={job.status} />
                </td>
              )}
              
              {displayColumns.includes(columns[6]) && (
                <td className="px-6 py-4 whitespace-nowrap">
                  <div className="flex items-center">
                    <div className="w-24">
                      <div className="progress-bar h-2">
                        <div
                          className="progress-bar-fill"
                          style={{ width: `${job.progress * 100}%` }}
                        />
                      </div>
                    </div>
                    <span className="ml-2 text-sm text-gray-600">
                      {(job.progress * 100).toFixed(0)}%
                    </span>
                  </div>
                </td>
              )}
              
              {displayColumns.includes(columns[7]) && (
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                  {formatDateTime(job.created_at)}
                </td>
              )}
              
              {displayColumns.includes(columns[8]) && (
                <td className="px-6 py-4 whitespace-nowrap text-sm font-medium">
                  <div className="flex items-center space-x-2">
                    <Link
                      to={`/jobs/${job.job_id}`}
                      className="text-primary-600 hover:text-primary-700"
                      title="상세보기"
                    >
                      <EyeIcon className="h-5 w-5" />
                    </Link>
                    
                    {job.status === JobStatus.COMPLETED && onRegenerate && (
                      <button
                        onClick={() => onRegenerate(job.job_id)}
                        className="text-blue-600 hover:text-blue-700"
                        title="재생성"
                      >
                        <ArrowPathIcon className="h-5 w-5" />
                      </button>
                    )}
                    
                    {(job.status === JobStatus.PENDING ||
                      job.status === JobStatus.PROCESSING) &&
                      onCancel && (
                        <button
                          onClick={() => onCancel(job.job_id)}
                          className="text-red-600 hover:text-red-700"
                          title="취소"
                        >
                          <XMarkIcon className="h-5 w-5" />
                        </button>
                      )}
                  </div>
                </td>
              )}
            </tr>
          ))}
          
          {sortedJobs.length === 0 && (
            <tr>
              <td
                colSpan={displayColumns.length}
                className="px-6 py-12 text-center text-gray-500"
              >
                작업이 없습니다
              </td>
            </tr>)}
       </tbody>
     </table>
   </div>
 );
};

export default JobsTable;