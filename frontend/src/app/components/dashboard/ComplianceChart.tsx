// ComplianceChart.tsx
'use client';
import { Card } from '../ui/Card';
import dynamic from 'next/dynamic';
import type { ApexOptions } from 'apexcharts';

// Dynamically import Chart.js to avoid SSR issues
const Chart = dynamic(() => import('react-apexcharts'), { ssr: false });

export const ComplianceChart = ({ results }: { results: any[] | null }) => {
  if (!results || results.length === 0) return null;
  
  const foundCount = results.filter(r => r.compliance_status === 'Found').length;
  const notFoundCount = results.filter(r => r.compliance_status === 'Not found').length;
  const errorCount = results.filter(r => r.compliance_status === 'Error').length;
  
  const chartOptions: ApexOptions = {
    chart: {
      type: "donut",
    },
    labels: ['Found', 'Not Found', 'Errors'],
    colors: ['#10B981', '#EF4444', '#F59E0B'],
    legend: {
      position: 'bottom',
    },
    dataLabels: {
      enabled: false,
    },
    plotOptions: {
      pie: {
        donut: {
          size: '65%',
          labels: {
            show: true,
            total: {
              show: true,
              label: 'Total',
              color: '#6B7280',
            }
          }
        }
      }
    },
  };
  
  const chartSeries = [foundCount, notFoundCount, errorCount];
  
  return (
    <Card title="Compliance Overview" className="mb-6">
      <div className="h-80">
        <Chart 
          options={chartOptions} 
          series={chartSeries} 
          type="donut" 
          height="100%" 
          width="100%" 
        />
      </div>
    </Card>
  );
};
