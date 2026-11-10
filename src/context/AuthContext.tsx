import React, { createContext, useContext, useEffect, useState } from 'react';
import { User, onAuthStateChanged } from 'firebase/auth';
import {
  auth,
  loginWithGoogle as fbLoginWithGoogle,
  loginAnonymously as fbLoginAnonymously,
  logoutUser as fbLogout,
  testFirestoreConnection,
  saveUserDataset as fbSaveDataset,
  loadUserDatasets as fbLoadDatasets,
  deleteUserDataset as fbDeleteDataset,
  StoredDataset,
} from '../utils/firebase';
import { DataDot } from '../types/ml';

interface AuthContextType {
  user: User | null;
  loading: boolean;
  loginWithGoogle: () => Promise<void>;
  loginAnonymously: () => Promise<void>;
  logout: () => Promise<void>;
  savedDatasets: StoredDataset[];
  refreshDatasets: () => Promise<void>;
  saveDataset: (dataset: {
    name: string;
    description: string;
    feature1: string;
    feature2: string;
    target?: string;
    dots: DataDot[];
  }) => Promise<string>;
  deleteDataset: (id: string) => Promise<void>;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const AuthProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [savedDatasets, setSavedDatasets] = useState<StoredDataset[]>([]);

  // Listen to Auth State
  useEffect(() => {
    testFirestoreConnection();

    const unsubscribe = onAuthStateChanged(auth, async (currentUser) => {
      setUser(currentUser);
      setLoading(false);
      if (currentUser) {
        try {
          const list = await fbLoadDatasets(currentUser.uid);
          setSavedDatasets(list);
        } catch (err) {
          console.error('Failed to load user datasets:', err);
        }
      } else {
        setSavedDatasets([]);
      }
    });

    return () => unsubscribe();
  }, []);

  const refreshDatasets = async () => {
    if (!user) return;
    try {
      const list = await fbLoadDatasets(user.uid);
      setSavedDatasets(list);
    } catch (err) {
      console.error('Failed to refresh datasets:', err);
    }
  };

  const loginWithGoogle = async () => {
    try {
      await fbLoginWithGoogle();
    } catch (err) {
      console.error('Google Sign-in failed:', err);
      throw err;
    }
  };

  const loginAnonymously = async () => {
    try {
      await fbLoginAnonymously();
    } catch (err) {
      console.error('Anonymous Sign-in failed:', err);
      throw err;
    }
  };

  const logout = async () => {
    try {
      await fbLogout();
    } catch (err) {
      console.error('Sign-out failed:', err);
      throw err;
    }
  };

  const saveDataset = async (dataset: {
    name: string;
    description: string;
    feature1: string;
    feature2: string;
    target?: string;
    dots: DataDot[];
  }): Promise<string> => {
    if (!user) {
      // Auto login anonymously to persist if not logged in
      const anonUser = await fbLoginAnonymously();
      const id = await fbSaveDataset(anonUser.uid, dataset);
      await refreshDatasets();
      return id;
    }
    const id = await fbSaveDataset(user.uid, dataset);
    await refreshDatasets();
    return id;
  };

  const deleteDataset = async (id: string): Promise<void> => {
    if (!user) return;
    await fbDeleteDataset(user.uid, id);
    await refreshDatasets();
  };

  return (
    <AuthContext.Provider
      value={{
        user,
        loading,
        loginWithGoogle,
        loginAnonymously,
        logout,
        savedDatasets,
        refreshDatasets,
        saveDataset,
        deleteDataset,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = (): AuthContextType => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};
