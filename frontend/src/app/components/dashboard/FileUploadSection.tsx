// FileUploadSection.tsx
'use client';
import { useState, useCallback, ChangeEvent } from 'react';
import { Button } from '../ui/Button';
import { Card } from '../ui/Card';
import { useFilePreview } from '@/app/hooks/useFilePreview';

export const FileUploadSection = ({ 
  onUpload,
  isLoading,
}: { 
  onUpload: (checklist: File, pdf: File, sheetName: string) => void;
  isLoading: boolean;
}) => {
  const [checklist, setChecklist] = useState<File | null>(null);
  const [pdf, setPdf] = useState<File | null>(null);
  const [sheetName, setSheetName] = useState('0');
  
  const { preview: excelPreview, sheetNames } = useFilePreview(checklist, 'excel');
  const { preview: pdfPreview } = useFilePreview(pdf, 'pdf');

  const handleChecklistChange = useCallback((e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files?.[0]) {
      setChecklist(e.target.files[0]);
      setSheetName('0'); // Reset to first sheet
    }
  }, []);

  const handlePdfChange = useCallback((e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files?.[0]) {
      setPdf(e.target.files[0]);
    }
  }, []);

  const handleSubmit = useCallback(() => {
    if (checklist && pdf) {
      onUpload(checklist, pdf, sheetName);
      
    }
  }, [checklist, pdf, sheetName, onUpload]);

  return (
    <Card title="Upload Documents" className="mb-6">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Excel Upload */}
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            SRA Checklist (Excel)
          </label>
          <div className="flex items-center justify-center w-full">
            <label className="flex flex-col items-center justify-center w-full h-64 border-2 border-gray-300 border-dashed rounded-lg cursor-pointer bg-gray-50 dark:border-gray-600 dark:bg-gray-700 hover:bg-gray-100 dark:hover:bg-gray-600">
              <div className="flex flex-col items-center justify-center pt-5 pb-6">
                <svg className="w-8 h-8 mb-4 text-gray-500 dark:text-gray-400" aria-hidden="true" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 20 16">
                  <path stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 13h3a3 3 0 0 0 0-6h-.025A5.56 5.56 0 0 0 16 6.5 5.5 5.5 0 0 0 5.207 5.021C5.137 5.017 5.071 5 5 5a4 4 0 0 0 0 8h2.167M10 15V6m0 0L8 8m2-2 2 2"/>
                </svg>
                <p className="mb-2 text-sm text-gray-500 dark:text-gray-400">
                  <span className="font-semibold">Click to upload</span> or drag and drop
                </p>
                <p className="text-xs text-gray-500 dark:text-gray-400">
                  Excel (.xlsx, .xls)
                </p>
              </div>
              <input 
                type="file" 
                className="hidden"
                accept=".xlsx,.xls"
                onChange={handleChecklistChange}
                disabled={isLoading}
              />
            </label>
          </div>
          
          {excelPreview && (
            <div className="mt-4">
              <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">Preview</h4>
              <div className="p-3 bg-gray-50 dark:bg-gray-700 rounded-md text-sm text-gray-600 dark:text-gray-300 whitespace-pre-wrap">
                {excelPreview}
              </div>
              
              {sheetNames.length > 0 && (
                <div className="mt-3">
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                    Select Sheet
                  </label>
                  <select
                    value={sheetName}
                    onChange={(e) => setSheetName(e.target.value)}
                    className="w-full p-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                    disabled={isLoading}
                  >
                    {sheetNames.map((name, index) => (
                      <option key={index} value={index}>{name}</option>
                    ))}
                  </select>
                </div>
              )}
            </div>
          )}
        </div>
        
        {/* PDF Upload */}
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            Financial Statement (PDF)
          </label>
          <div className="flex items-center justify-center w-full">
            <label className="flex flex-col items-center justify-center w-full h-64 border-2 border-gray-300 border-dashed rounded-lg cursor-pointer bg-gray-50 dark:border-gray-600 dark:bg-gray-700 hover:bg-gray-100 dark:hover:bg-gray-600">
              <div className="flex flex-col items-center justify-center pt-5 pb-6">
                <svg className="w-8 h-8 mb-4 text-gray-500 dark:text-gray-400" aria-hidden="true" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 20 16">
                  <path stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 13h3a3 3 0 0 0 0-6h-.025A5.56 5.56 0 0 0 16 6.5 5.5 5.5 0 0 0 5.207 5.021C5.137 5.017 5.071 5 5 5a4 4 0 0 0 0 8h2.167M10 15V6m0 0L8 8m2-2 2 2"/>
                </svg>
                <p className="mb-2 text-sm text-gray-500 dark:text-gray-400">
                  <span className="font-semibold">Click to upload</span> or drag and drop
                </p>
                <p className="text-xs text-gray-500 dark:text-gray-400">
                  PDF (.pdf)
                </p>
              </div>
              <input 
                type="file" 
                className="hidden"
                accept=".pdf"
                onChange={handlePdfChange}
                disabled={isLoading}
              />
            </label>
          </div>
          
          {pdfPreview && (
            <div className="mt-4">
              <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">Preview</h4>
              <div className="p-3 bg-gray-50 dark:bg-gray-700 rounded-md">
                <img 
                  src={pdfPreview} 
                  alt="PDF preview" 
                  className="max-h-40 mx-auto"
                />
              </div>
            </div>
          )}
        </div>
      </div>
      
      <div className="mt-6 flex justify-end">
        <Button
          variant="primary"
          onClick={handleSubmit}
          disabled={!checklist || !pdf || isLoading}
          isLoading={isLoading}
        >
          Start Compliance Check
        </Button>
      </div>
    </Card>
  );
};






