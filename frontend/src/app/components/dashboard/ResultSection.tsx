// ResultsSection.tsx
'use client';
import { useState, useMemo } from 'react';
import { Card } from '../ui/Card';
import { Badge } from '../ui/Badge';
import { Button } from '../ui/Button';

export const ResultsSection = ({ 
  results, 
  jobId,
  onNewCheck
}: { 
  results: any[] | null;
  jobId: string | null;
  onNewCheck: () => void;
}) => {
  const [filter, setFilter] = useState<'all' | 'found' | 'not_found'>('all');
  const [searchQuery, setSearchQuery] = useState('');
  const [expandedRow, setExpandedRow] = useState<string | null>(null);
  
  const filteredResults = useMemo(() => {
    if (!results) return [];
    
    let filtered = results;
    
    // Apply status filter
    if (filter === 'found') {
      filtered = filtered.filter(r => r.compliance_status === 'Found');
    } else if (filter === 'not_found') {
      filtered = filtered.filter(r => r.compliance_status === 'Not found');
    }
    
    // Apply search filter
    if (searchQuery) {
      const query = searchQuery.toLowerCase();
      filtered = filtered.filter(r => 
        r.requirement.toLowerCase().includes(query) ||
        (r.main_requirement && r.main_requirement.toLowerCase().includes(query)) ||
        r.category.toLowerCase().includes(query)
      );
    }
    
    return filtered;
  }, [results, filter, searchQuery]);
  
  const handleDownload = () => {
    if (jobId) {
      window.open(`http://localhost:8000/download/${jobId}`, '_blank');
    }
  };
  
  if (!results) return null;
  
  return (
    <Card 
      title="Compliance Results"
      actions={
        <div className="flex space-x-2">
          <input
            type="text"
            placeholder="Search requirements..."
            className="px-3 py-1.5 border border-gray-300 dark:border-gray-600 rounded-md text-sm bg-white dark:bg-gray-800 text-gray-900 dark:text-white"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
          />
          <Button 
            variant={filter === 'all' ? 'primary' : 'outline'}
            onClick={() => setFilter('all')}
          >
            All ({results.length})
          </Button>
          <Button 
            variant={filter === 'found' ? 'primary' : 'outline'}
            onClick={() => setFilter('found')}
          >
            Found ({results.filter(r => r.compliance_status === 'Found').length})
          </Button>
          <Button 
            variant={filter === 'not_found' ? 'primary' : 'outline'}
            onClick={() => setFilter('not_found')}
          >
            Not Found ({results.filter(r => r.compliance_status === 'Not found').length})
          </Button>
        </div>
      }
    >
      <div className="overflow-x-auto">
        <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
          <thead className="bg-gray-50 dark:bg-gray-800">
            <tr>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Requirement</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Category</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Status</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Page</th>
            </tr>
          </thead>
          <tbody className="bg-white dark:bg-gray-800 divide-y divide-gray-200 dark:divide-gray-700">
            {filteredResults.map((result) => (
              <>
                <tr 
                  key={result.id} 
                  className="hover:bg-gray-50 dark:hover:bg-gray-750 cursor-pointer"
                  onClick={() => setExpandedRow(expandedRow === result.id ? null : result.id)}
                >
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="text-sm font-medium text-gray-900 dark:text-white">
                      {result.requirement}
                    </div>
                    {!result.is_main && result.main_requirement && (
                      <div className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                        {result.main_requirement}
                      </div>
                    )}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">
                    {result.category}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <Badge 
                      variant={
                        result.compliance_status === 'Found' ? 'success' : 
                        result.compliance_status === 'Not found' ? 'error' : 'warning'
                      }
                    >
                      {result.compliance_status}
                    </Badge>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">
                    {result.page || '-'}
                  </td>
                </tr>
                {expandedRow === result.id && (
                  <tr className="bg-gray-50 dark:bg-gray-750">
                    <td colSpan={4} className="px-6 py-4">
                      <div className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">Context Evidence:</div>
                      <div className="bg-gray-100 dark:bg-gray-700 p-4 rounded-md text-sm text-gray-600 dark:text-gray-300 whitespace-pre-wrap max-h-60 overflow-y-auto">
                        {result.context || 'No context available'}
                      </div>
                    </td>
                  </tr>
                )}
              </>
            ))}
          </tbody>
        </table>
      </div>
      
      {filteredResults.length === 0 && (
        <div className="text-center py-12">
          <p className="text-gray-500 dark:text-gray-400">No results found</p>
        </div>
      )}
      
      <div className="mt-8 flex justify-between">
        <Button variant="outline" onClick={onNewCheck}>
          Start New Check
        </Button>
        <Button variant="primary" onClick={handleDownload}>
          Download Full Report
        </Button>
      </div>
    </Card>
  );
};