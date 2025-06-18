'use client';
import { useState, useEffect } from 'react';
import { Sidebar } from './components/layout/Sidebar';
import { Topbar } from '@/app/components/layout/Topbar';
import { FileUploadSection } from '@/app/components/dashboard/FileUploadSection';
import { ProgressSection } from '@/app/components/dashboard/ProgressSection';
import { SummaryCards } from '@/app/components/dashboard/SummaryCards';
import { ComplianceChart } from '@/app/components/dashboard/ComplianceChart';
import { ResultsSection } from './components/dashboard/ResultSection';
import { uploadFiles, getJobStatus, getResults } from '@/app/utils/api';
import { useTheme } from '@/app/context/ThemeContext';
import { ThemeProvider } from '@/app/context/ThemeContext';

const HomePageContent = () => {
  const [jobId, setJobId] = useState<string | null>(null);
  const [status, setStatus] = useState<any>(null);
  const [results, setResults] = useState<any[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const { theme } = useTheme();

  const handleUpload = async (checklist: File, pdf: File, sheetName: string) => {
    try {
      setIsUploading(true);
      setError(null);
      setResults(null);
      const response = await uploadFiles(checklist, pdf, sheetName);
      setJobId(response.job_id);
      setStatus({
        job_id: response.job_id,
        status: 'processing',
        progress: 0,
        message: 'Processing started...'
      });
    } catch (err) {
      setError('Failed to start processing. Please try again.');
      console.error(err);
    } finally {
      setIsUploading(false);
    }
  };

  useEffect(() => {
    let intervalId: NodeJS.Timeout | null = null;

    const fetchStatus = async () => {
      if (!jobId) return;
      
      try {
        const statusData = await getJobStatus(jobId);
        setStatus(statusData);
        
        if (statusData.status === 'completed') {
          const resultsData = await getResults(jobId);
          setResults(resultsData.results);
          if (intervalId) clearInterval(intervalId);
        }
        
        if (statusData.status === 'error') {
          setError(statusData.message || 'Processing error occurred');
          if (intervalId) clearInterval(intervalId);
        }
      } catch (err) {
        setError('Failed to fetch status. Please try again.');
        console.error(err);
        if (intervalId) clearInterval(intervalId);
      }
    };

    if (jobId && status?.status !== 'completed' && status?.status !== 'error') {
      intervalId = setInterval(fetchStatus, 2000);
    }

    return () => {
      if (intervalId) clearInterval(intervalId);
    };
  }, [jobId, status]);

  const handleNewCheck = () => {
    setJobId(null);
    setStatus(null);
    setResults(null);
    setError(null);
  };

  return (
    <div className={`min-h-screen ${theme === 'dark' ? 'dark bg-gray-900' : 'bg-gray-50'}`}>
      <Sidebar />
      <div className="md:pl-64 flex flex-col flex-1">
        <Topbar />
        <main className="flex-1 pb-8">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="py-8">
              <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Compliance Checker</h1>
              <p className="mt-2 text-sm text-gray-700 dark:text-gray-300">
                Automatically verify financial statements against SRA requirements
              </p>
            </div>
            
            {!jobId && !results && (
              <FileUploadSection 
                onUpload={handleUpload} 
                isLoading={isUploading} 
              />
            )}
            
            {error && (
              <div className="mb-6 bg-red-50 dark:bg-red-900/30 border border-red-200 dark:border-red-700 text-red-700 dark:text-red-200 px-4 py-3 rounded-md">
                {error}
              </div>
            )}
            
            {status && <ProgressSection status={status} />}
            
            {results && (
              <>
                <SummaryCards results={results} />
                <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
                  <div className="lg:col-span-2">
                    <ComplianceChart results={results} />
                  </div>
                  <div className="bg-gradient-to-br from-blue-600 to-blue-700 rounded-xl p-6 text-white">
                    <h3 className="text-lg font-medium mb-4">Compliance Insights</h3>
                    <p className="mb-4">
                      Based on our analysis of {results.length} requirements, we found:
                    </p>
                    <ul className="space-y-2">
                      <li className="flex items-center">
                        <div className="w-3 h-3 bg-green-400 rounded-full mr-2"></div>
                        <span>
                          {results.filter(r => r.compliance_status === 'Found').length} compliant items
                        </span>
                      </li>
                      <li className="flex items-center">
                        <div className="w-3 h-3 bg-red-400 rounded-full mr-2"></div>
                        <span>
                          {results.filter(r => r.compliance_status === 'Not found').length} non-compliant items
                        </span>
                      </li>
                      <li className="flex items-center">
                        <div className="w-3 h-3 bg-yellow-400 rounded-full mr-2"></div>
                        <span>
                          {results.filter(r => r.compliance_status === 'Error').length} items with errors
                        </span>
                      </li>
                    </ul>
                    <div className="mt-6">
                      <h4 className="font-medium mb-2">Recommendations</h4>
                      <p>
                        Focus on addressing the non-compliant items first. Pay special attention to requirements 
                        in the "Accounting Policies" category, as they often have the highest impact on compliance.
                      </p>
                    </div>
                  </div>
                </div>
                <ResultsSection 
                  results={results} 
                  jobId={jobId} 
                  onNewCheck={handleNewCheck} 
                />
              </>
            )}
          </div>
        </main>
      </div>
    </div>
  );
};

export default function Home() {
  return (
    <ThemeProvider>
      <HomePageContent />
    </ThemeProvider>
  );
}