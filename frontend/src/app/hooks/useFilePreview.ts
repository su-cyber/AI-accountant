import { useState, useEffect } from 'react';

export const useFilePreview = (file: File | null, type: 'excel' | 'pdf') => {
  const [preview, setPreview] = useState<string | null>(null);
  const [sheetNames, setSheetNames] = useState<string[]>([]);

  useEffect(() => {
    let reader: FileReader | null = null;

    if (!file) {
      setPreview(null);
      setSheetNames([]);
      return;
    }

    if (type === 'pdf') {
      // Generate PDF preview
      reader = new FileReader();
      reader.onload = () => {
        if (reader) {
          setPreview(reader.result as string);
        }
      };
      reader.readAsDataURL(file);
    } else {
      // Generate Excel preview and extract sheet names
      reader = new FileReader();
      reader.onload = (e) => {
        const data = e.target?.result;
        if (data) {
          // Extract sheet names (simplified - in a real app you'd use a library)
          const sheets = Array(5).fill(0).map((_, i) => `Sheet ${i + 1}`);
          setSheetNames(sheets);
          
          // Create a simple preview
          const previewText = `Excel File: ${file.name}\nSheets: ${sheets.join(', ')}`;
          setPreview(previewText);
        }
      };
      reader.readAsText(file);
    }

    return () => {
      if (reader && reader.readyState === 1) {
        reader.abort();
      }
    };
  }, [file, type]);

  return { preview, sheetNames };
};