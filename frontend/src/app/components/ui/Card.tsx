// Card.tsx
import { ReactNode } from 'react';
import clsx from 'clsx';

export const Card = ({
  children,
  className,
  title,
  actions,
}: {
  children: ReactNode;
  className?: string;
  title?: string;
  actions?: ReactNode;
}) => {
  return (
    <div className={clsx(
      "bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 overflow-hidden",
      className
    )}>
      {(title || actions) && (
        <div className="px-6 py-4 border-b border-gray-200 dark:border-gray-700 flex items-center justify-between">
          {title && <h3 className="text-lg font-medium text-gray-900 dark:text-white">{title}</h3>}
          {actions}
        </div>
      )}
      <div className="p-6">
        {children}
      </div>
    </div>
  );
};