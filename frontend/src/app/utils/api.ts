import axios from 'axios';

const API_URL = 'http://localhost:8000';

export const uploadFiles = async (checklist: File, pdf: File, sheetName: string) => {
  const formData = new FormData();
  formData.append('checklist', checklist);
  formData.append('pdf', pdf);
  
  const response = await axios.post(`${API_URL}/upload`, formData, {
    headers: { 'Content-Type': 'multipart/form-data' }
  });
  
  return response.data;
};

export const getJobStatus = async (jobId: string) => {
  const response = await axios.get(`${API_URL}/status/${jobId}`);
  return response.data;
};

export const getResults = async (jobId: string) => {
  const response = await axios.get(`${API_URL}/results/${jobId}`);
  return response.data;
};

export const downloadReport = (jobId: string) => {
  window.open(`${API_URL}/download/${jobId}`, '_blank');
};