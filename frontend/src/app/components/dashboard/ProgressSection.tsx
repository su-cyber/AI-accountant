// ProgressSection.tsx
'use client';
import { Card } from '../ui/Card';
import { Badge } from '../ui/Badge';

export const ProgressSection = ({ status }: { status: any }) => {
  if (!status) return null;
  
  const statusColors = {
    processing: 'info',
    completed: 'success',
    error: 'error',
  };
  
  return (
    <Card title="Processing Status" className="mb-6">
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <div>
            <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300">Status</h4>
            <div className="mt-1 flex items-center">
              <Badge variant={statusColors[status.status as keyof typeof statusColors] as any}>
                {status.status}
              </Badge>
              <span className="ml-2 text-sm text-gray-500 dark:text-gray-400">
                {status.message}
              </span>
            </div>
          </div>
          <span className="text-xl font-semibold text-blue-600 dark:text-blue-400">
            {Math.round(status.progress)}%
          </span>
        </div>
        
        <div className="w-full bg-gray-200 rounded-full h-2.5 dark:bg-gray-700">
          <div 
            className="bg-blue-600 h-2.5 rounded-full transition-all duration-300" 
            style={{ width: `${status.progress}%` }}
          ></div>
        </div>
        
        <div className="grid grid-cols-2 gap-4">
          <div>
            <p className="text-sm text-gray-500 dark:text-gray-400">Requirements Processed</p>
            <p className="text-lg font-semibold">
              {status.processed_requirements || 0} / {status.total_requirements || 0}
            </p>
          </div>
          <div>
            <p className="text-sm text-gray-500 dark:text-gray-400">Checks Found</p>
            <p className="text-lg font-semibold text-green-600 dark:text-green-400">
              {status.found_count || 0}
            </p>
          </div>
        </div>
      </div>
    </Card>
  );
};