import { DataDot } from '../types/ml';

export interface ParsedDataset {
  name: string;
  description: string;
  columns: string[];
  numericColumns: string[];
  rawRows: Array<Record<string, any>>;
  feature1: string;
  feature2: string;
  labelColumn: string;
  dots: DataDot[];
}

export interface ParseOptions {
  feature1Col?: string;
  feature2Col?: string;
  labelCol?: string;
  usePCA?: boolean;
}

/**
 * Universal Client-Side Data Parser
 * Supports: CSV, TSV, JSON (objects/arrays), Python lists, Markdown tables, Whitespace-separated text
 */
export function parseRawData(rawInput: string, options?: ParseOptions): ParsedDataset {
  const trimmed = rawInput.trim();
  if (!trimmed) {
    throw new Error('Data input is empty. Please paste or upload your dataset.');
  }

  let columns: string[] = [];
  let rawRows: Array<Record<string, any>> = [];

  // 1. Try parsing as JSON
  if (trimmed.startsWith('{') || trimmed.startsWith('[')) {
    try {
      const parsed = JSON.parse(trimmed);

      if (Array.isArray(parsed)) {
        if (parsed.length > 0) {
          if (typeof parsed[0] === 'object' && !Array.isArray(parsed[0])) {
            // Array of objects [{ x: 1, y: 2, label: 0 }]
            rawRows = parsed;
            columns = Object.keys(parsed[0]);
          } else if (Array.isArray(parsed[0])) {
            // Array of arrays [[1, 2, 0], [3, 4, 1]]
            const colCount = parsed[0].length;
            columns = Array.from({ length: colCount }, (_, i) => `Col_${i + 1}`);
            rawRows = parsed.map((arr) => {
              const row: Record<string, any> = {};
              columns.forEach((col, idx) => {
                row[col] = arr[idx];
              });
              return row;
            });
          }
        }
      } else if (typeof parsed === 'object') {
        // Object containing arrays { "features": [[...], ...], "labels": [...] }
        if (Array.isArray(parsed.data)) {
          return parseRawData(JSON.stringify(parsed.data), options);
        } else if (Array.isArray(parsed.dots)) {
          // DotML native format
          const dots = parsed.dots.map((d: any, i: number) => ({
            id: d.id || `dot-${i}`,
            x: Math.max(0.02, Math.min(0.98, Number(d.x) || 0.5)),
            y: Math.max(0.02, Math.min(0.98, Number(d.y) || 0.5)),
            label: d.label !== undefined ? (Number(d.label) >= 1 ? 1 : 0) : 0,
            continuousTarget: d.continuousTarget !== undefined ? Number(d.continuousTarget) : d.y,
          }));
          return {
            name: parsed.datasetName || 'Imported Dataset',
            description: parsed.description || 'Imported from JSON',
            columns: ['x', 'y', 'label'],
            numericColumns: ['x', 'y'],
            rawRows: dots,
            feature1: parsed.feature1Name || 'Feature 1',
            feature2: parsed.feature2Name || 'Feature 2',
            labelColumn: parsed.targetName || 'Target',
            dots,
          };
        }
      }
    } catch {
      // Not valid JSON, proceed to tabular / text parsing
    }
  }

  // 2. Python list string format (e.g. [[1.2, 3.4], [5.6, 7.8]])
  if (rawRows.length === 0 && trimmed.startsWith('[') && trimmed.endsWith(']')) {
    try {
      const sanitized = trimmed
        .replace(/'/g, '"')
        .replace(/None/g, 'null')
        .replace(/True/g, 'true')
        .replace(/False/g, 'false');
      const pyParsed = JSON.parse(sanitized);
      if (Array.isArray(pyParsed) && pyParsed.length > 0 && Array.isArray(pyParsed[0])) {
        const colCount = pyParsed[0].length;
        columns = Array.from({ length: colCount }, (_, i) => `Feature_${i + 1}`);
        rawRows = pyParsed.map((arr) => {
          const row: Record<string, any> = {};
          columns.forEach((col, idx) => {
            row[col] = arr[idx];
          });
          return row;
        });
      }
    } catch {
      // Fall through to text delimiter parsing
    }
  }

  // 3. Tabular parsing (CSV, TSV, Markdown table, pipe, semicolon, whitespace)
  if (rawRows.length === 0) {
    const lines = trimmed.split(/\r?\n/).filter((l) => l.trim().length > 0);
    if (lines.length < 2) {
      throw new Error('Dataset must contain at least 2 rows of data.');
    }

    // Detect delimiter
    const firstLine = lines[0];
    let delimiter = ',';
    if (firstLine.includes('\t')) delimiter = '\t';
    else if (firstLine.includes(';') && !firstLine.includes(',')) delimiter = ';';
    else if (firstLine.includes('|')) delimiter = '|';
    else if (!firstLine.includes(',') && /\s{2,}|\s+/.test(firstLine)) delimiter = ' ';

    // Helper to split row by delimiter
    const splitRow = (rowStr: string): string[] => {
      if (delimiter === ' ') {
        return rowStr.trim().split(/\s+/);
      }
      if (delimiter === '|') {
        return rowStr
          .split('|')
          .map((c) => c.trim())
          .filter((_, idx, arr) => idx > 0 && idx < arr.length - 1);
      }
      // Standard CSV parsing with quote handling
      const tokens: string[] = [];
      let inQuotes = false;
      let currentToken = '';

      for (let i = 0; i < rowStr.length; i++) {
        const char = rowStr[i];
        if (char === '"' || char === "'") {
          inQuotes = !inQuotes;
        } else if (char === delimiter && !inQuotes) {
          tokens.push(currentToken.trim());
          currentToken = '';
        } else {
          currentToken += char;
        }
      }
      tokens.push(currentToken.trim());
      return tokens;
    };

    // Check if line 0 looks like header (contains non-numeric strings)
    const headerTokens = splitRow(lines[0]);
    const secondLineTokens = splitRow(lines[1]);

    const isHeaderLine = headerTokens.some((t) => isNaN(Number(t)) && t.length > 0);

    let startLine = 0;
    if (isHeaderLine) {
      columns = headerTokens.map((h, i) => h.replace(/["']/g, '') || `Column_${i + 1}`);
      startLine = 1;
      // Skip Markdown divider like |---|---|
      if (lines[1] && lines[1].includes('---')) {
        startLine = 2;
      }
    } else {
      columns = secondLineTokens.map((_, i) => `Feature_${i + 1}`);
    }

    for (let i = startLine; i < lines.length; i++) {
      const lineStr = lines[i].trim();
      if (!lineStr || lineStr.includes('---')) continue;
      const tokens = splitRow(lineStr);
      if (tokens.length === 0) continue;

      const row: Record<string, any> = {};
      columns.forEach((col, idx) => {
        const val = tokens[idx];
        if (val === undefined || val === '') {
          row[col] = null;
        } else if (!isNaN(Number(val))) {
          row[col] = Number(val);
        } else {
          row[col] = val.replace(/["']/g, '').trim();
        }
      });
      rawRows.push(row);
    }
  }

  if (rawRows.length === 0) {
    throw new Error('Could not parse any records from the provided input.');
  }

  // Find numeric columns
  const numericColumns = columns.filter((col) => {
    let numCount = 0;
    let totalCount = 0;
    rawRows.forEach((r) => {
      if (r[col] !== null && r[col] !== undefined) {
        totalCount++;
        if (typeof r[col] === 'number' || !isNaN(Number(r[col]))) {
          numCount++;
        }
      }
    });
    return totalCount > 0 && numCount / totalCount >= 0.7;
  });

  if (numericColumns.length < 2) {
    throw new Error(
      `Dataset needs at least 2 numerical columns for 2D visualization. Found numeric columns: ${
        numericColumns.join(', ') || 'None'
      }`
    );
  }

  // Pick default features
  let f1 = options?.feature1Col || numericColumns[0];
  let f2 = options?.feature2Col || numericColumns[1] || numericColumns[0];
  if (!numericColumns.includes(f1)) f1 = numericColumns[0];
  if (!numericColumns.includes(f2)) f2 = numericColumns[1] || numericColumns[0];

  // Pick default label column (last non-feature column or last column)
  const nonFeatureCols = columns.filter((c) => c !== f1 && c !== f2);
  let labelCol =
    options?.labelCol ||
    nonFeatureCols.find((c) => /label|target|class|species|type|outcome/i.test(c)) ||
    nonFeatureCols[nonFeatureCols.length - 1] ||
    '';

  // Build Dots with Normalization
  let dots: DataDot[] = [];

  // Check if PCA on all numeric columns was requested
  if (options?.usePCA && numericColumns.length > 2) {
    dots = projectHighDimensionalTo2D(rawRows, numericColumns, labelCol);
  } else {
    // Normal Min-Max scaling for f1 and f2
    const f1Vals = rawRows.map((r) => Number(r[f1])).filter((v) => !isNaN(v));
    const f2Vals = rawRows.map((r) => Number(r[f2])).filter((v) => !isNaN(v));

    const min1 = Math.min(...f1Vals);
    const max1 = Math.max(...f1Vals);
    const min2 = Math.min(...f2Vals);
    const max2 = Math.max(...f2Vals);

    const range1 = max1 - min1 || 1;
    const range2 = max2 - min2 || 1;

    // Detect unique classes if labelCol exists
    const uniqueLabels = Array.from(
      new Set(
        rawRows
          .map((r) => r[labelCol])
          .filter((v) => v !== undefined && v !== null && v !== '')
      )
    );

    dots = rawRows.map((row, idx) => {
      const v1 = Number(row[f1]);
      const v2 = Number(row[f2]);

      // Normalize to [0.06, 0.94]
      const nx = 0.06 + 0.88 * ((isNaN(v1) ? 0.5 : v1 - min1) / range1);
      const ny = 0.06 + 0.88 * ((isNaN(v2) ? 0.5 : v2 - min2) / range2);

      let label = 0;
      let continuousTarget = ny;

      if (labelCol && row[labelCol] !== undefined) {
        const rawLabelVal = row[labelCol];
        if (typeof rawLabelVal === 'number') {
          label = rawLabelVal >= 1 ? 1 : 0;
          continuousTarget = rawLabelVal;
        } else {
          // Categorical mapping
          const str = String(rawLabelVal).toLowerCase().trim();
          if (str === '1' || str === 'true' || str === 'yes' || str === 'positive' || str === 'high' || str === 'malignant') {
            label = 1;
          } else if (uniqueLabels.length === 2) {
            label = uniqueLabels.indexOf(rawLabelVal) === 1 ? 1 : 0;
          } else {
            label = (idx % 2);
          }
        }
      } else {
        // If no label, classify by position or unlabelled
        label = nx + ny > 1 ? 1 : 0;
      }

      return {
        id: `dot-${idx + 1}`,
        x: Math.max(0.04, Math.min(0.96, nx)),
        y: Math.max(0.04, Math.min(0.96, ny)),
        label,
        continuousTarget,
      };
    });
  }

  return {
    name: 'Custom Dataset',
    description: `Parsed ${dots.length} data dots with features ${f1} and ${f2}`,
    columns,
    numericColumns,
    rawRows,
    feature1: f1,
    feature2: f2,
    labelColumn: labelCol,
    dots,
  };
}

/**
 * Projects high-dimensional tabular data down to 2 principal components
 */
function projectHighDimensionalTo2D(
  rows: Array<Record<string, any>>,
  numericCols: string[],
  labelCol?: string
): DataDot[] {
  const n = rows.length;
  const d = numericCols.length;

  // 1. Matrix X [n x d]
  const matrix: number[][] = [];
  rows.forEach((r) => {
    const rowVec = numericCols.map((c) => Number(r[c]) || 0);
    matrix.push(rowVec);
  });

  // 2. Mean center
  const means: number[] = Array(d).fill(0);
  for (let j = 0; j < d; j++) {
    for (let i = 0; i < n; i++) {
      means[j] += matrix[i][j];
    }
    means[j] /= n;
  }

  const centered = matrix.map((row) => row.map((val, j) => val - means[j]));

  // 3. Power iteration for PC1 & PC2
  let v1 = Array(d).fill(1 / Math.sqrt(d));
  for (let iter = 0; iter < 20; iter++) {
    let nextV = Array(d).fill(0);
    for (let i = 0; i < n; i++) {
      let dot = 0;
      for (let j = 0; j < d; j++) dot += centered[i][j] * v1[j];
      for (let j = 0; j < d; j++) nextV[j] += centered[i][j] * dot;
    }
    let norm = Math.hypot(...nextV) || 1;
    v1 = nextV.map((v) => v / norm);
  }

  // Deflate for PC2
  const deflated = centered.map((row) => {
    let dot = 0;
    for (let j = 0; j < d; j++) dot += row[j] * v1[j];
    return row.map((val, j) => val - dot * v1[j]);
  });

  let v2 = Array(d).fill(1 / Math.sqrt(d));
  for (let iter = 0; iter < 20; iter++) {
    let nextV = Array(d).fill(0);
    for (let i = 0; i < n; i++) {
      let dot = 0;
      for (let j = 0; j < d; j++) dot += deflated[i][j] * v2[j];
      for (let j = 0; j < d; j++) nextV[j] += deflated[i][j] * dot;
    }
    let norm = Math.hypot(...nextV) || 1;
    v2 = nextV.map((v) => v / norm);
  }

  // Project points
  const scores1 = centered.map((row) => row.reduce((sum, val, j) => sum + val * v1[j], 0));
  const scores2 = centered.map((row) => row.reduce((sum, val, j) => sum + val * v2[j], 0));

  const min1 = Math.min(...scores1);
  const max1 = Math.max(...scores1);
  const min2 = Math.min(...scores2);
  const max2 = Math.max(...scores2);

  const range1 = max1 - min1 || 1;
  const range2 = max2 - min2 || 1;

  return rows.map((r, i) => {
    const nx = 0.08 + 0.84 * ((scores1[i] - min1) / range1);
    const ny = 0.08 + 0.84 * ((scores2[i] - min2) / range2);
    let label = 0;
    if (labelCol && r[labelCol] !== undefined) {
      label = Number(r[labelCol]) >= 1 ? 1 : 0;
    } else {
      label = i % 2;
    }
    return {
      id: `dot-${i + 1}`,
      x: Math.max(0.04, Math.min(0.96, nx)),
      y: Math.max(0.04, Math.min(0.96, ny)),
      label,
      continuousTarget: ny,
    };
  });
}

// ----------------------------------------------------
// 5 REAL-WORLD CLASSIC ML BENCHMARK DATASETS
// ----------------------------------------------------
export interface BenchmarkDataset {
  id: string;
  name: string;
  category: string;
  description: string;
  rawText: string;
}

export const BENCHMARK_DATASETS: BenchmarkDataset[] = [
  {
    id: 'iris',
    name: "Fisher's Iris Flowers",
    category: 'Botanical Classification',
    description: 'Sepal length vs Petal length classifying Iris Setosa vs Versicolor species.',
    rawText: `sepal_length,sepal_width,petal_length,petal_width,species
5.1,3.5,1.4,0.2,setosa
4.9,3.0,1.4,0.2,setosa
4.7,3.2,1.3,0.2,setosa
4.6,3.1,1.5,0.2,setosa
5.0,3.6,1.4,0.2,setosa
5.4,3.9,1.7,0.4,setosa
4.6,3.4,1.4,0.3,setosa
5.0,3.4,1.5,0.2,setosa
4.4,2.9,1.4,0.2,setosa
4.9,3.1,1.5,0.1,setosa
5.4,3.7,1.5,0.2,setosa
4.8,3.4,1.6,0.2,setosa
4.8,3.0,1.4,0.1,setosa
4.3,3.0,1.1,0.1,setosa
5.8,4.0,1.2,0.2,setosa
5.7,4.4,1.5,0.4,setosa
5.4,3.9,1.3,0.4,setosa
5.1,3.5,1.4,0.3,setosa
5.7,3.8,1.7,0.3,setosa
5.1,3.8,1.5,0.3,setosa
7.0,3.2,4.7,1.4,versicolor
6.4,3.2,4.5,1.5,versicolor
6.9,3.1,4.9,1.5,versicolor
5.5,2.3,4.0,1.3,versicolor
6.5,2.8,4.6,1.5,versicolor
5.7,2.8,4.5,1.3,versicolor
6.3,3.3,4.7,1.6,versicolor
4.9,2.4,3.3,1.0,versicolor
6.6,2.9,4.6,1.3,versicolor
5.2,2.7,3.9,1.4,versicolor
5.0,2.0,3.5,1.0,versicolor
5.9,3.0,4.2,1.5,versicolor
6.0,2.2,4.0,1.0,versicolor
6.1,2.9,4.7,1.4,versicolor
5.6,2.9,3.6,1.3,versicolor
6.7,3.1,4.4,1.4,versicolor
5.6,3.0,4.5,1.5,versicolor
5.8,2.7,4.1,1.0,versicolor
6.2,2.2,4.5,1.5,versicolor
5.6,2.5,3.9,1.1,versicolor`,
  },
  {
    id: 'customers',
    name: 'Customer Income & Spending',
    category: 'Retail Clustering',
    description: 'Annual Income ($k) vs Spending Score (1-100) identifying high-value buyer personas.',
    rawText: `CustomerID,AnnualIncome_k,SpendingScore_1_100,HighSpender
1,15,39,0
2,15,81,1
3,16,6,0
4,16,77,1
5,17,40,0
6,17,76,1
7,18,6,0
8,18,94,1
9,19,3,0
10,19,72,1
11,28,82,1
12,28,40,0
13,38,35,0
14,39,61,1
15,40,55,1
16,42,52,0
17,43,54,0
18,48,50,0
19,54,47,0
20,54,59,1
21,70,56,1
22,71,35,0
23,73,5,0
24,75,93,1
25,78,16,0
26,78,90,1
27,81,5,0
28,85,97,1
29,87,13,0
30,88,69,1
31,93,14,0
32,97,32,0
33,99,97,1
34,103,23,0
35,103,85,1
36,113,91,1
37,120,79,1
38,126,28,0
39,137,18,0
40,137,83,1`,
  },
  {
    id: 'exams',
    name: 'Student Study Hours & Sleep',
    category: 'Academic Pass/Fail',
    description: 'Weekly study hours and hours of sleep determining exam outcome.',
    rawText: `study_hours,sleep_hours,exam_passed
2.5,6.0,0
3.0,5.5,0
3.8,7.0,0
4.2,6.5,0
5.0,7.2,1
5.5,8.0,1
6.0,6.5,1
6.8,7.5,1
7.2,8.0,1
8.0,7.0,1
2.0,8.5,0
2.8,4.5,0
4.0,5.0,0
4.5,7.8,0
5.8,5.2,1
6.2,8.5,1
7.0,6.0,1
7.5,7.5,1
8.5,8.0,1
9.0,7.2,1
1.5,6.5,0
2.2,7.0,0
3.5,6.2,0
4.8,6.8,1
5.2,7.4,1
6.5,7.1,1
7.8,8.2,1
8.2,6.8,1
9.5,7.5,1
3.2,5.0,0`,
  },
  {
    id: 'housing',
    name: 'Housing Size vs Distance',
    category: 'Real Estate Valuation',
    description: 'Square meters and distance to city center (km) predicting high vs standard price tier.',
    rawText: `sqm,distance_km,price_tier
45,18.5,0
52,15.2,0
60,12.0,0
68,14.5,0
75,9.2,1
82,7.5,1
90,5.0,1
110,4.2,1
125,3.0,1
140,2.1,1
40,22.0,0
48,19.0,0
55,16.5,0
70,11.0,0
85,8.0,1
95,6.5,1
105,4.8,1
120,3.5,1
135,2.5,1
155,1.8,1
50,20.0,0
58,17.2,0
64,13.8,0
78,10.5,1
88,7.8,1
100,5.5,1
115,4.0,1
130,2.8,1
145,2.0,1
160,1.2,1`,
  },
  {
    id: 'sensor',
    name: 'IoT Sensor Vibration & Temp',
    category: 'Anomaly Detection',
    description: 'Vibration frequency (Hz) vs Motor Temperature (°C) detecting machine anomalies.',
    rawText: `vibration_hz,temp_celsius,anomaly
22.4,42.1,0
23.1,43.5,0
21.9,41.8,0
24.0,44.2,0
22.8,42.9,0
23.5,43.8,0
21.5,41.2,0
24.2,44.7,0
22.1,42.5,0
23.8,44.0,0
48.5,78.2,1
51.2,82.4,1
49.8,79.5,1
52.0,85.1,1
47.9,77.0,1
22.0,42.0,0
23.2,43.1,0
22.6,42.7,0
24.1,44.5,0
21.8,41.5,0
50.5,81.0,1
53.1,86.2,1
49.2,78.8,1
22.7,42.8,0
23.4,43.6,0`,
  },
];
