// SummaryCards.tsx
'use client';
import { Card } from '../ui/Card';

export const SummaryCards = ({ results }: { results: any[] | null }) => {
  if (!results || results.length === 0) return null;
  
  const foundCount = results.filter(r => r.compliance_status === 'Found').length;
  const notFoundCount = results.filter(r => r.compliance_status === 'Not found').length;
  const totalCount = results.length;
  const complianceRate = totalCount > 0 ? Math.round((foundCount / totalCount) * 100) : 0;
  
  return (
    <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-6">
      <Card className="bg-gradient-to-r from-blue-500 to-blue-600 text-white">
        <h3 className="text-sm font-medium opacity-80">Total Requirements</h3>
        <p className="text-3xl font-bold mt-2">{totalCount}</p>
      </Card>
      
      <Card className="bg-gradient-to-r from-green-500 to-green-600 text-white">
        <h3 className="text-sm font-medium opacity-80">Compliance Found</h3>
        <p className="text-3xl font-bold mt-2">{foundCount}</p>
      </Card>
      
      <Card className="bg-gradient-to-r from-red-500 to-red-600 text-white">
        <h3 className="text-sm font-medium opacity-80">Not Found</h3>
        <p className="text-3xl font-bold mt-2">{notFoundCount}</p>
      </Card>
      
      <Card className="bg-gradient-to-r from-purple-500 to-purple-600 text-white">
        <h3 className="text-sm font-medium opacity-80">Compliance Rate</h3>
        <p className="text-3xl font-bold mt-2">{complianceRate}%</p>
      </Card>
    </div>
  );
};