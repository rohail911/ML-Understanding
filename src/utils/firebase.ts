import { initializeApp, getApps, getApp } from 'firebase/app';
import {
  getAuth,
  signInWithPopup,
  GoogleAuthProvider,
  signOut,
  onAuthStateChanged,
  signInAnonymously,
  User,
} from 'firebase/auth';
import {
  getFirestore,
  doc,
  setDoc,
  getDoc,
  getDocFromServer,
  collection,
  query,
  getDocs,
  deleteDoc,
  serverTimestamp,
} from 'firebase/firestore';
import firebaseConfig from '../../firebase-applet-config.json';
import { DataDot } from '../types/ml';

// Initialize Firebase App
const app = getApps().length === 0 ? initializeApp(firebaseConfig) : getApp();

// Initialize Auth
export const auth = getAuth(app);
export const googleProvider = new GoogleAuthProvider();
googleProvider.setCustomParameters({ prompt: 'select_account' });

// Initialize Firestore (with explicit database ID from config if provided)
export const db = firebaseConfig.firestoreDatabaseId
  ? getFirestore(app, firebaseConfig.firestoreDatabaseId)
  : getFirestore(app);

// Connection test on boot
export async function testFirestoreConnection(): Promise<boolean> {
  try {
    await getDocFromServer(doc(db, 'test', 'connection'));
    return true;
  } catch (error: any) {
    if (error instanceof Error && error.message.includes('the client is offline')) {
      console.warn('Firebase client is offline or network restricted.');
      return false;
    }
    // Permissions error on non-existent test doc is normal and means server responded
    return true;
  }
}

// Auth helpers
export async function loginWithGoogle(): Promise<User> {
  const result = await signInWithPopup(auth, googleProvider);
  // Store or update profile
  const user = result.user;
  try {
    const userRef = doc(db, 'users', user.uid);
    await setDoc(
      userRef,
      {
        userId: user.uid,
        email: user.email || '',
        displayName: user.displayName || 'ML Explorer',
        photoURL: user.photoURL || '',
        lastLogin: serverTimestamp(),
      },
      { merge: true }
    );
  } catch (err) {
    console.error('Failed to update user profile in Firestore:', err);
  }
  return user;
}

export async function loginAnonymously(): Promise<User> {
  const result = await signInAnonymously(auth);
  return result.user;
}

export async function logoutUser(): Promise<void> {
  await signOut(auth);
}

// Firestore Dataset Persistence
export interface StoredDataset {
  id: string;
  userId: string;
  name: string;
  description: string;
  feature1: string;
  feature2: string;
  target?: string;
  dotCount: number;
  dots: DataDot[];
  createdAt?: any;
}

export async function saveUserDataset(
  userId: string,
  dataset: {
    name: string;
    description: string;
    feature1: string;
    feature2: string;
    target?: string;
    dots: DataDot[];
  }
): Promise<string> {
  const datasetId = `ds-${Date.now()}`;
  const datasetRef = doc(db, 'users', userId, 'datasets', datasetId);

  await setDoc(datasetRef, {
    id: datasetId,
    userId,
    name: dataset.name,
    description: dataset.description,
    feature1: dataset.feature1,
    feature2: dataset.feature2,
    target: dataset.target || '',
    dotCount: dataset.dots.length,
    dotsJson: JSON.stringify(dataset.dots),
    createdAt: new Date().toISOString(),
  });

  return datasetId;
}

export async function loadUserDatasets(userId: string): Promise<StoredDataset[]> {
  const colRef = collection(db, 'users', userId, 'datasets');
  const snap = await getDocs(query(colRef));
  const list: StoredDataset[] = [];

  snap.forEach((d) => {
    const data = d.data();
    let dots: DataDot[] = [];
    try {
      dots = JSON.parse(data.dotsJson || '[]');
    } catch {
      dots = [];
    }
    list.push({
      id: d.id,
      userId: data.userId,
      name: data.name,
      description: data.description,
      feature1: data.feature1,
      feature2: data.feature2,
      target: data.target,
      dotCount: data.dotCount,
      dots,
      createdAt: data.createdAt,
    });
  });

  return list;
}

export async function deleteUserDataset(userId: string, datasetId: string): Promise<void> {
  const datasetRef = doc(db, 'users', userId, 'datasets', datasetId);
  await deleteDoc(datasetRef);
}

// Firestore Experiment Persistence
export interface StoredExperiment {
  id: string;
  userId: string;
  algorithmId: string;
  algorithmName: string;
  datasetName: string;
  metrics: Record<string, any>;
  params: Record<string, any>;
  createdAt?: string;
}

export async function saveExperiment(
  userId: string,
  experiment: {
    algorithmId: string;
    algorithmName: string;
    datasetName: string;
    metrics: Record<string, any>;
    params: Record<string, any>;
  }
): Promise<string> {
  const expId = `exp-${Date.now()}`;
  const expRef = doc(db, 'users', userId, 'experiments', expId);

  await setDoc(expRef, {
    id: expId,
    userId,
    algorithmId: experiment.algorithmId,
    algorithmName: experiment.algorithmName,
    datasetName: experiment.datasetName,
    metricsJson: JSON.stringify(experiment.metrics),
    paramsJson: JSON.stringify(experiment.params),
    createdAt: new Date().toISOString(),
  });

  return expId;
}

export async function loadUserExperiments(userId: string): Promise<StoredExperiment[]> {
  const colRef = collection(db, 'users', userId, 'experiments');
  const snap = await getDocs(query(colRef));
  const list: StoredExperiment[] = [];

  snap.forEach((d) => {
    const data = d.data();
    let metrics = {};
    let params = {};
    try {
      metrics = JSON.parse(data.metricsJson || '{}');
      params = JSON.parse(data.paramsJson || '{}');
    } catch {
      // fallback
    }
    list.push({
      id: d.id,
      userId: data.userId,
      algorithmId: data.algorithmId,
      algorithmName: data.algorithmName,
      datasetName: data.datasetName,
      metrics,
      params,
      createdAt: data.createdAt,
    });
  });

  return list;
}
