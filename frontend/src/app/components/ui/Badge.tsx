import clsx from 'clsx';

export const Badge = ({
  variant = 'default',
  children,
}: {
  variant?: 'default' | 'success' | 'warning' | 'error' | 'info';
  children: React.ReactNode;
}) => {
  const variantClasses = {
    default: 'bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-200',
    success: 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-100',
    warning: 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-100',
    error: 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-100',
    info: 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-100',
  };
  
  return (
    <span className={clsx(
      "inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium",
      variantClasses[variant]
    )}>
      {children}
    </span>
  );
};