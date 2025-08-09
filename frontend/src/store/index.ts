import { create } from 'zustand';
import { devtools, persist } from 'zustand/middleware';
import {
  AppState,
  User,
  Notification,
  AppSettings,
  Domain,
  DataType,
  OutputFormat,
} from '../types';

interface StoreState extends AppState {
  // Actions
  initialize: () => Promise<void>;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  setUser: (user: User | null) => void;
  addNotification: (notification: Omit<Notification, 'id' | 'timestamp' | 'read'>) => void;
  markNotificationAsRead: (id: string) => void;
  clearNotifications: () => void;
  updateSettings: (settings: Partial<AppSettings>) => void;
  reset: () => void;
}

const initialSettings: AppSettings = {
  theme: 'light',
  language: 'ko',
  notifications: {
    email: true,
    push: true,
    inApp: true,
  },
  defaults: {
    domain: Domain.GENERAL,
    dataType: DataType.TEXT,
    outputFormat: OutputFormat.JSON,
    quantity: 100,
  },
};

export const useStore = create<StoreState>()(
  devtools(
    persist(
      (set, get) => ({
        // Initial State
        isInitialized: false,
        isLoading: false,
        error: null,
        user: null,
        notifications: [],
        settings: initialSettings,

        // Actions
        initialize: async () => {
          try {
            set({ isLoading: true });
            
            // 저장된 설정 로드
            const savedSettings = localStorage.getItem('app-settings');
            if (savedSettings) {
              const parsedSettings = JSON.parse(savedSettings);
              set({ settings: { ...initialSettings, ...parsedSettings } });
            }

            // 사용자 정보 로드 (토큰이 있는 경우)
            const token = localStorage.getItem('access_token');
            if (token) {
              // API 호출하여 사용자 정보 가져오기
              // const userInfo = await api.auth.getMe();
              // set({ user: userInfo });
            }

            set({ isInitialized: true, isLoading: false });
          } catch (error) {
            console.error('Initialization failed:', error);
            set({ error: 'Failed to initialize app', isLoading: false });
          }
        },

        setLoading: (loading) => set({ isLoading: loading }),

        setError: (error) => set({ error }),

        setUser: (user) => set({ user }),

        addNotification: (notification) => {
          const newNotification: Notification = {
            ...notification,
            id: Date.now().toString(),
            timestamp: new Date(),
            read: false,
          };

          set((state) => ({
            notifications: [newNotification, ...state.notifications].slice(0, 50), // 최대 50개 유지
          }));

          // 인앱 알림이 활성화된 경우 토스트 표시
          if (get().settings.notifications.inApp) {
            import('react-hot-toast').then(({ toast }) => {
              const toastFn = notification.type === 'error' ? toast.error 
                : notification.type === 'success' ? toast.success
                : notification.type === 'warning' ? toast.error
                : toast;
              
              toastFn(notification.title);
            });
          }
        },

        markNotificationAsRead: (id) => {
          set((state) => ({
            notifications: state.notifications.map((n) =>
              n.id === id ? { ...n, read: true } : n
            ),
          }));
        },

        clearNotifications: () => set({ notifications: [] }),

        updateSettings: (newSettings) => {
          set((state) => {
            const updatedSettings = {
              ...state.settings,
              ...newSettings,
            };

            // 로컬 스토리지에 저장
            localStorage.setItem('app-settings', JSON.stringify(updatedSettings));

            // 테마 적용
            if (newSettings.theme) {
              document.documentElement.classList.remove('light', 'dark');
              if (newSettings.theme === 'auto') {
                const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
                document.documentElement.classList.add(prefersDark ? 'dark' : 'light');
              } else {
                document.documentElement.classList.add(newSettings.theme);
              }
            }

            return { settings: updatedSettings };
          });
        },

        reset: () => {
          localStorage.removeItem('access_token');
          localStorage.removeItem('app-settings');
          set({
            isInitialized: false,
            isLoading: false,
            error: null,
            user: null,
            notifications: [],
            settings: initialSettings,
          });
        },
      }),
      {
        name: 'app-store',
        partialize: (state) => ({
          settings: state.settings,
          notifications: state.notifications,
        }),
      }
    )
  )
);

// Hooks
export const useUser = () => useStore((state) => state.user);
export const useSettings = () => useStore((state) => state.settings);
export const useNotifications = () => useStore((state) => state.notifications);
export const useLoading = () => useStore((state) => state.isLoading);