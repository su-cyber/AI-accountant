export interface ComplianceResult {
  id: string;
  requirement: string;
  category: string;
  is_main: boolean;
  compliance_status: string;
  page?: number;
  context?: string;
  main_requirement?: string;
}

export interface ProcessingStatus {
  job_id: string;
  status: string;
  progress: number;
  message: string;
  total_requirements?: number;
  processed_requirements?: number;
  found_count?: number;
}