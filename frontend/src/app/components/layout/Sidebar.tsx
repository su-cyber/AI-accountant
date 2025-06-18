// Sidebar.tsx
'use client';
import { useTheme } from '@/app/context/ThemeContext';
import { Button } from '../ui/Button';
import { 
  HomeIcon, 
  DocumentTextIcon, 
  ChartBarIcon, 
  CogIcon, 
  MoonIcon, 
  SunIcon 
} from '@heroicons/react/24/outline';

export const Sidebar = () => {
  const { theme, toggleTheme } = useTheme();
  
  const navigation = [
    { name: 'Dashboard', href: '#', icon: HomeIcon, current: true },
    { name: 'Reports', href: '#', icon: DocumentTextIcon, current: false },
    { name: 'Analytics', href: '#', icon: ChartBarIcon, current: false },
    { name: 'Settings', href: '#', icon: CogIcon, current: false },
  ];
  
  return (
    <div className="hidden md:flex md:w-64 md:flex-col md:fixed md:inset-y-0">
      <div className="flex-1 flex flex-col min-h-0 bg-white dark:bg-gray-800 border-r border-gray-200 dark:border-gray-700">
        <div className="flex items-center justify-center h-16 flex-shrink-0 px-4 bg-gradient-to-r from-blue-600 to-blue-700">
          <div className="text-white text-xl font-bold">ComplianceAI</div>
        </div>
        <div className="flex-1 flex flex-col overflow-y-auto py-4">
          <nav className="flex-1 px-2 space-y-1">
            {navigation.map((item) => (
              <a
                key={item.name}
                href={item.href}
                className={`${
                  item.current
                    ? 'bg-blue-100 text-blue-600 dark:bg-blue-900/50 dark:text-blue-300'
                    : 'text-gray-700 hover:bg-gray-100 dark:text-gray-300 dark:hover:bg-gray-700'
                } group flex items-center px-4 py-3 text-sm font-medium rounded-md`}
              >
                <item.icon
                  className={`${
                    item.current
                      ? 'text-blue-500 dark:text-blue-400'
                      : 'text-gray-500 group-hover:text-gray-700 dark:text-gray-400 dark:group-hover:text-gray-300'
                  } mr-3 flex-shrink-0 h-6 w-6`}
                  aria-hidden="true"
                />
                {item.name}
              </a>
            ))}
          </nav>
          
          <div className="px-4 mt-auto">
            <Button
              variant="ghost"
              onClick={toggleTheme}
              className="w-full justify-start"
            >
              {theme === 'dark' ? (
                <>
                  <SunIcon className="h-5 w-5 mr-3" />
                  Light Mode
                </>
              ) : (
                <>
                  <MoonIcon className="h-5 w-5 mr-3" />
                  Dark Mode
                </>
              )}
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
};

