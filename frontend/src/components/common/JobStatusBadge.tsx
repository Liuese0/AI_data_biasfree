import React from 'react';
import { JobStatus } from '../../types';

interface JobStatusBadgeProps {
  status: JobStatus | string;
}

const JobStatusBadge: React.FC<JobStatusBadgeProps> = ({ status }) => {
  const getStatusColor = () => {
    switch (status) {
      case JobStatus.PENDING:
        return 'bg-gray-100 text-gray-800';
      case JobStatus.PROCESSING:
        return 'bg-blue-100 text-blue-800';
      case JobStatus.COMPLETED:
        return 'bg-green-100 text-green-800';
      case JobStatus.VALIDATED:
        return 'bg-purple-100 text-purple-800';
      case JobStatus.FAILED:
        return 'bg-red-100 text-red-800';
      case JobStatus.CANCELLED:
        return 'bg-yellow-100 text-yellow-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  const getStatusText = () => {
    switch (status) {
      case JobStatus.PENDING:
        return '대기중';
      case JobStatus.PROCESSING:
        return '처리중';
      case JobStatus.COMPLETED:
        return '완료';
      case JobStatus.VALIDATED:
        return '검증완료';
      case JobStatus.FAILED:
        return '실패';
      case JobStatus.CANCELLED:
        return '취소됨';
      default:
        return status;
    }
  };

  return (
    <span
      className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${getStatusColor()}`}
    >
      {getStatusText()}
    </span>
  );
};

export default JobStatusBadge;