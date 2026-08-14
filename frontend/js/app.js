/**
 * MediSynth.AI — Main Application
 */
const state = { datasetId: null, jobId: null, federationId: null, loading: false };

// ── Navigation ──
function switchTab(tabId) {
  document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));
  document.getElementById(tabId)?.classList.add('active');
  document.querySelector(`[data-tab="${tabId}"]`)?.classList.add('active');
}

// ── Toast ──
function toast(msg, type = 'success') {
  const c = document.getElementById('toast-container');
  const t = document.createElement('div');
  t.className = `toast ${type}`;
  t.innerHTML = `<span>${type === 'success' ? '✓' : type === 'error' ? '✗' : '⚠'}</span><span>${msg}</span>`;
  c.appendChild(t);
  setTimeout(() => t.remove(), 4000);
}

// ── Loading ──
function showLoading(msg) {
  state.loading = true;
  document.getElementById('loading-overlay').style.display = 'flex';
  document.getElementById('loading-text').textContent = msg || 'Processing...';
}
function hideLoading() {
  state.loading = false;
  document.getElementById('loading-overlay').style.display = 'none';
}

// ── Upload ──
async function handleFileUpload(file) {
  if (!file || !file.name.endsWith('.csv')) { toast('Please upload a CSV file', 'error'); return; }
  showLoading('Uploading and analyzing data...');
  try {
    const res = await api.uploadFile(file);
    state.datasetId = res.data.dataset_id;
    renderDatasetInfo(res.data);
    toast(`Dataset uploaded: ${res.data.num_rows} rows, ${res.data.num_cols} columns`);
    switchTab('tab-generate');
  } catch (e) { toast(e.message, 'error'); }
  hideLoading();
}

async function loadSampleData() {
  showLoading('Loading sample healthcare dataset...');
  try {
    const res = await api.loadSample();
    state.datasetId = res.data.dataset_id;
    renderDatasetInfo(res.data);
    toast(`Sample data loaded: ${res.data.num_rows} rows`);
    switchTab('tab-generate');
  } catch (e) { toast(e.message, 'error'); }
  hideLoading();
}

function renderDatasetInfo(data) {
  document.getElementById('dataset-status').innerHTML = `
    <div class="grid grid-4">
      <div class="stat-card"><div class="stat-label">Dataset</div><div class="stat-value blue">${data.filename.substring(0, 15)}</div></div>
      <div class="stat-card"><div class="stat-label">Rows</div><div class="stat-value green">${data.num_rows.toLocaleString()}</div></div>
      <div class="stat-card"><div class="stat-label">Columns</div><div class="stat-value purple">${data.num_cols}</div></div>
      <div class="stat-card"><div class="stat-label">ID</div><div class="stat-value blue" style="font-size:0.85rem">${data.dataset_id}</div></div>
    </div>
    <div class="card" style="margin-top:1rem"><div class="card-title">📋 Data Preview</div>
    <div style="overflow-x:auto"><table class="data-table"><thead><tr>${data.columns.map(c => `<th>${c}</th>`).join('')}</tr></thead>
    <tbody>${(data.preview || []).map(r => `<tr>${data.columns.map(c => `<td>${r[c] ?? ''}</td>`).join('')}</tr>`).join('')}</tbody></table></div></div>
    <div class="card"><div class="card-title">📊 Column Types</div>
    <div class="grid grid-4">${Object.entries(data.column_types).map(([k, v]) =>
      `<div class="stat-card"><div class="stat-label">${k}</div><span class="badge badge-${v === 'numerical' ? 'info' : v === 'boolean' ? 'purple' : 'warning'}">${v}</span></div>`
    ).join('')}</div></div>`;
  document.getElementById('gen-dataset-id').value = data.dataset_id;
}

// ── Generation ──
async function generateData() {
  const dsId = document.getElementById('gen-dataset-id').value || state.datasetId;
  if (!dsId) { toast('Upload data first', 'error'); return; }
  const model = document.getElementById('gen-model').value || 'statistical';
  const msgs = {
    'statistical': 'Generating synthetic data (this should be quick)...',
    'ctgan': 'Training CTGAN model... This takes 2-10 minutes on CPU. Please wait.',
    'tvae': 'Training TVAE model... This takes 1-5 minutes on CPU. Please wait.',
  };
  showLoading(msgs[model] || 'Processing...');
  try {
    const params = {
      dataset_id: dsId,
      num_rows: parseInt(document.getElementById('gen-rows').value) || 1000,
      model_type: document.getElementById('gen-model').value || 'statistical',
      epochs: parseInt(document.getElementById('gen-epochs').value) || 100,
      epsilon: parseFloat(document.getElementById('gen-epsilon').value) || 1.0,
      delta: parseFloat(document.getElementById('gen-delta').value) || 1e-5,
      dp_mechanism: document.getElementById('gen-mechanism').value || 'gaussian',
      apply_dp: document.getElementById('gen-apply-dp').checked,
    };
    const res = await api.generate(params);
    state.jobId = res.data.job_id;
    renderGenResult(res.data);
    toast(`Generated ${res.data.num_rows_generated} synthetic rows`);
  } catch (e) { toast(e.message, 'error'); }
  hideLoading();
}

function renderGenResult(data) {
  document.getElementById('gen-result').innerHTML = `
    <div class="grid grid-4">
      <div class="stat-card"><div class="stat-label">Rows Generated</div><div class="stat-value green">${data.num_rows_generated}</div></div>
      <div class="stat-card"><div class="stat-label">Model</div><div class="stat-value blue">${data.model_type.toUpperCase()}</div></div>
      <div class="stat-card"><div class="stat-label">Training Time</div><div class="stat-value amber">${data.training_time_seconds}s</div></div>
      <div class="stat-card"><div class="stat-label">DP Applied</div><div class="stat-value ${data.dp_applied ? 'green' : 'red'}">${data.dp_applied ? '✓ Yes' : '✗ No'}</div></div>
    </div>
    ${data.dp_metadata ? `<div class="card" style="margin-top:1rem"><div class="card-title">🔐 Differential Privacy Details</div>
      <div class="grid grid-3">
        <div class="stat-card"><div class="stat-label">ε (Epsilon)</div><div class="stat-value amber">${data.dp_metadata.epsilon_actual}</div></div>
        <div class="stat-card"><div class="stat-label">δ (Delta)</div><div class="stat-value blue">${data.dp_metadata.delta_requested}</div></div>
        <div class="stat-card"><div class="stat-label">Mechanism</div><div class="stat-value purple">${data.dp_metadata.mechanism}</div></div>
      </div></div>` : ''}
    ${data.privacy_budget ? `<div class="card"><div class="card-title">💰 Privacy Budget</div>
      <div class="grid grid-3">
        <div class="stat-card"><div class="stat-label">Budget Used</div><div class="stat-value ${data.privacy_budget.utilization_pct > 75 ? 'red' : 'green'}">${data.privacy_budget.utilization_pct}%</div></div>
        <div class="stat-card"><div class="stat-label">ε Remaining</div><div class="stat-value blue">${data.privacy_budget.remaining_epsilon}</div></div>
        <div class="stat-card"><div class="stat-label">Queries</div><div class="stat-value purple">${data.privacy_budget.num_queries}</div></div>
      </div>
      <div class="progress-bar" style="margin-top:1rem"><div class="progress-fill ${data.privacy_budget.utilization_pct > 75 ? 'danger' : 'blue'}" style="width:${data.privacy_budget.utilization_pct}%"></div></div></div>` : ''}
    <div style="margin-top:1rem"><a href="${api.downloadUrl(data.job_id)}" class="btn btn-success" download>⬇ Download Synthetic CSV</a></div>`;
}

// ── Statistical Validation ──
async function runStatValidation() {
  if (!state.datasetId) { toast('Upload data first', 'error'); return; }
  showLoading('Running statistical validation...');
  try {
    const res = await api.validateStatistical({ dataset_id: state.datasetId, synthetic_job_id: state.jobId });
    renderStatResults(res.data);
    toast(`Quality Score: ${res.data.overall_quality_score}/100 (${res.data.quality_grade})`);
  } catch (e) { toast(e.message, 'error'); }
  hideLoading();
}

function renderStatResults(data) {
  const el = document.getElementById('stat-results');
  let chartsHtml = '';
  let idx = 0;
  for (const [col, report] of Object.entries(data.column_reports)) {
    if (report.distribution && (report.distribution.bins || report.distribution.categories)) {
      chartsHtml += `<div class="card"><div class="chart-container"><canvas id="dist-chart-${idx}"></canvas></div></div>`;
    }
    idx++;
  }
  el.innerHTML = `
    <div class="grid grid-3">
      <div class="stat-card"><div class="stat-label">Quality Score</div><div class="stat-value blue">${data.overall_quality_score}</div></div>
      <div class="stat-card"><div class="stat-label">Grade</div><div class="grade grade-${data.quality_grade}">${data.quality_grade}</div></div>
      <div class="stat-card"><div class="stat-label">Correlation MAE</div><div class="stat-value amber">${data.correlation?.mean_absolute_error?.toFixed(4) || 'N/A'}</div></div>
    </div>
    <div class="card" style="margin-top:1rem"><div class="card-title">📊 Per-Column KS Test</div>
    <table class="data-table"><thead><tr><th>Column</th><th>Type</th><th>KS / χ² Statistic</th><th>P-Value</th><th>Similar?</th><th>Score</th></tr></thead>
    <tbody>${Object.entries(data.column_reports).map(([col, r]) => {
      const test = r.ks_test || r.chi_squared || {};
      return `<tr><td>${col}</td><td><span class="badge badge-${r.type === 'numerical' ? 'info' : 'warning'}">${r.type}</span></td>
      <td>${test.statistic?.toFixed(4) ?? '-'}</td><td>${test.p_value?.toFixed(4) ?? '-'}</td>
      <td>${test.similar ? '<span class="badge badge-success">Yes</span>' : '<span class="badge badge-danger">No</span>'}</td>
      <td>${r.quality_score?.toFixed(1)}</td></tr>`;
    }).join('')}</tbody></table></div>
    ${data.correlation?.columns?.length > 1 ? `
    <div class="grid grid-2">
      <div class="card"><div class="card-title">🔵 Real Correlation</div><div id="heatmap-real" class="heatmap-container"></div></div>
      <div class="card"><div class="card-title">🟣 Synthetic Correlation</div><div id="heatmap-synth" class="heatmap-container"></div></div>
    </div>` : ''}
    <div class="grid grid-2">${chartsHtml}</div>`;

  // Render heatmaps
  if (data.correlation?.columns?.length > 1) {
    setTimeout(() => {
      charts.heatmapHTML('heatmap-real', data.correlation.real_correlation, data.correlation.columns);
      charts.heatmapHTML('heatmap-synth', data.correlation.synth_correlation, data.correlation.columns);
    }, 100);
  }
  // Render distribution charts
  setTimeout(() => {
    let i = 0;
    for (const [col, report] of Object.entries(data.column_reports)) {
      if (report.distribution) {
        if (report.type === 'numerical' && report.distribution.bins) charts.distribution(`dist-chart-${i}`, report.distribution, col);
        else if (report.distribution.categories) charts.categoricalDist(`dist-chart-${i}`, report.distribution, col);
      }
      i++;
    }
  }, 200);
}

// ── ML Validation ──
async function runMLValidation() {
  if (!state.datasetId) { toast('Upload data first', 'error'); return; }
  showLoading('Running ML utility validation (training 4 models)...');
  try {
    const target = document.getElementById('ml-target').value || undefined;
    const res = await api.validateML({ dataset_id: state.datasetId, synthetic_job_id: state.jobId, target_column: target || undefined });
    renderMLResults(res.data);
    toast(`Utility Score: ${res.data.utility_score}/100 (${res.data.utility_grade})`);
  } catch (e) { toast(e.message, 'error'); }
  hideLoading();
}

function renderMLResults(data) {
  const el = document.getElementById('ml-results');
  const r = data.results;
  el.innerHTML = `
    <div class="grid grid-4">
      <div class="stat-card"><div class="stat-label">Utility Score</div><div class="stat-value blue">${data.utility_score}</div></div>
      <div class="stat-card"><div class="stat-label">Grade</div><div class="grade grade-${data.utility_grade}">${data.utility_grade}</div></div>
      <div class="stat-card"><div class="stat-label">Accuracy Gap</div><div class="stat-value amber">${data.utility_gaps.accuracy_gap ?? '-'}</div></div>
      <div class="stat-card"><div class="stat-label">AUC Gap</div><div class="stat-value purple">${data.utility_gaps.auc_gap ?? '-'}</div></div>
    </div>
    <div class="card" style="margin-top:1rem"><div class="card-title">📊 Model Comparison</div>
    <table class="data-table"><thead><tr><th>Metric</th><th>Real (RF)</th><th>Synth (RF)</th><th>Real (GB)</th><th>Synth (GB)</th></tr></thead>
    <tbody>
      ${['accuracy', 'f1_score', 'roc_auc', 'precision', 'recall'].map(m => `<tr><td><strong>${m}</strong></td>
        <td>${r.trtr_rf[m] ?? '-'}</td><td>${r.tstr_rf[m] ?? '-'}</td>
        <td>${r.trtr_gb[m] ?? '-'}</td><td>${r.tstr_gb[m] ?? '-'}</td></tr>`).join('')}
    </tbody></table></div>
    <div class="grid grid-2">
      <div class="card"><div class="card-title">📊 Accuracy Comparison</div><div class="chart-container"><canvas id="ml-comparison-chart"></canvas></div></div>
      <div class="card"><div class="card-title">📈 ROC Curve</div><div class="chart-container"><canvas id="ml-roc-chart"></canvas></div></div>
    </div>`;
  setTimeout(() => {
    charts.comparisonBar('ml-comparison-chart', ['Accuracy', 'F1', 'AUC'],
      [r.trtr_rf.accuracy, r.trtr_rf.f1_score, r.trtr_rf.roc_auc],
      [r.tstr_rf.accuracy, r.tstr_rf.f1_score, r.tstr_rf.roc_auc]);
    if (r.tstr_rf.roc_curve) charts.rocCurve('ml-roc-chart', r.tstr_rf.roc_curve.fpr, r.tstr_rf.roc_curve.tpr, 'TSTR ROC');
  }, 200);
}

// ── Attack Simulation ──
async function runAttacks() {
  if (!state.datasetId) { toast('Upload data first', 'error'); return; }
  showLoading('Simulating privacy attacks...');
  try {
    const res = await api.simulateAttacks({ dataset_id: state.datasetId, synthetic_job_id: state.jobId });
    renderAttackResults(res.data);
    toast(`Overall Risk: ${res.data.overall_risk_score}/100 (${res.data.overall_risk_level})`);
  } catch (e) { toast(e.message, 'error'); }
  hideLoading();
}

function renderAttackResults(data) {
  const a = data.attacks;
  const rl = l => `risk-${l}`;
  document.getElementById('attack-results').innerHTML = `
    <div class="grid grid-4">
      <div class="stat-card"><div class="stat-label">Overall Risk</div><div class="stat-value ${rl(data.overall_risk_level)}">${data.overall_risk_score}</div></div>
      <div class="stat-card"><div class="stat-label">MIA Risk</div><div class="stat-value ${rl(a.membership_inference.risk_level)}">${a.membership_inference.risk_score}</div></div>
      <div class="stat-card"><div class="stat-label">Re-ID Risk</div><div class="stat-value ${rl(a.reidentification.risk_level)}">${a.reidentification.risk_score}</div></div>
      <div class="stat-card"><div class="stat-label">Attr Inference</div><div class="stat-value ${rl(a.attribute_inference.risk_level)}">${a.attribute_inference.risk_score}</div></div>
    </div>
    <div class="grid grid-2">
      <div class="card"><div class="card-title">🎯 Threat Radar</div><div class="chart-container"><canvas id="attack-radar"></canvas></div></div>
      <div class="card"><div class="card-title">📈 MIA ROC Curve</div><div class="chart-container"><canvas id="attack-roc"></canvas></div></div>
    </div>
    <div class="card"><div class="card-title">🔍 Attack Details</div>
    <table class="data-table"><thead><tr><th>Attack</th><th>Metric</th><th>Value</th><th>Risk Level</th></tr></thead>
    <tbody>
      <tr><td>Membership Inference</td><td>AUC</td><td>${a.membership_inference.attack_auc}</td><td><span class="badge badge-${a.membership_inference.risk_level === 'low' ? 'success' : 'danger'}">${a.membership_inference.risk_level}</span></td></tr>
      <tr><td>Re-Identification</td><td>Records at Risk</td><td>${a.reidentification.records_at_risk_pct}%</td><td><span class="badge badge-${a.reidentification.risk_level === 'low' ? 'success' : 'danger'}">${a.reidentification.risk_level}</span></td></tr>
      <tr><td>Attribute Inference</td><td>Avg Advantage</td><td>${a.attribute_inference.average_advantage}</td><td><span class="badge badge-${a.attribute_inference.risk_level === 'low' ? 'success' : 'danger'}">${a.attribute_inference.risk_level}</span></td></tr>
    </tbody></table></div>
    <div class="card"><p>${data.summary}</p></div>`;
  setTimeout(() => {
    charts.radar('attack-radar', data.radar_chart.labels, data.radar_chart.risk_scores);
    if (a.membership_inference.roc_curve) charts.rocCurve('attack-roc', a.membership_inference.roc_curve.fpr, a.membership_inference.roc_curve.tpr, 'MIA');
  }, 200);
}

// ── Federated Learning ──
async function createFederation() {
  showLoading('Creating federation...');
  try {
    const rounds = parseInt(document.getElementById('fed-rounds').value) || 5;
    const res = await api.createFederation({ total_rounds: rounds });
    state.federationId = res.data.federation_id;
    document.getElementById('fed-id-display').textContent = state.federationId;
    document.getElementById('fed-controls').style.display = 'block';
    toast(`Federation created: ${state.federationId}`);
  } catch (e) { toast(e.message, 'error'); }
  hideLoading();
}

async function addHospital() {
  const file = document.getElementById('fed-hospital-file').files[0];
  const name = document.getElementById('fed-hospital-name').value || 'Hospital';
  if (!file || !state.federationId) { toast('Select a file and create federation first', 'error'); return; }
  showLoading('Adding hospital data...');
  try {
    const res = await api.addHospital(state.federationId, name, file);
    toast(`Hospital "${name}" added: ${res.data.num_records} records`);
    const list = document.getElementById('fed-hospital-list');
    list.innerHTML += `<div class="stat-card"><div class="stat-label">${name}</div><div class="stat-value green">${res.data.num_records}</div></div>`;
  } catch (e) { toast(e.message, 'error'); }
  hideLoading();
}

async function runFederatedTraining() {
  if (!state.federationId) { toast('Create federation first', 'error'); return; }
  showLoading('Running federated training across hospitals...');
  try {
    const eps = parseFloat(document.getElementById('fed-epsilon').value) || 1.0;
    const res = await api.federatedTrain({ federation_id: state.federationId, dp_epsilon: eps, apply_dp: true });
    document.getElementById('fed-results').innerHTML = `
      <div class="grid grid-4">
        <div class="stat-card"><div class="stat-label">Rounds</div><div class="stat-value blue">${res.data.rounds_completed}</div></div>
        <div class="stat-card"><div class="stat-label">Hospitals</div><div class="stat-value green">${res.data.num_hospitals}</div></div>
        <div class="stat-card"><div class="stat-label">Order Independent</div><div class="stat-value ${res.data.order_independent_verified ? 'green' : 'red'}">${res.data.order_independent_verified ? '✓' : '✗'}</div></div>
        <div class="stat-card"><div class="stat-label">DP ε</div><div class="stat-value amber">${res.data.dp_epsilon}</div></div>
      </div>`;
    toast(`Federated training complete: ${res.data.rounds_completed} rounds`);
  } catch (e) { toast(e.message, 'error'); }
  hideLoading();
}

async function federatedGenerate() {
  if (!state.federationId) return;
  showLoading('Generating from federated model...');
  try {
    const rows = parseInt(document.getElementById('fed-gen-rows').value) || 1000;
    const res = await api.federatedGenerate({ federation_id: state.federationId, num_rows: rows });
    toast(`Generated ${res.data.num_rows} federated synthetic rows`);
    document.getElementById('fed-gen-result').innerHTML = `<div class="card"><div class="card-title">✅ Generated ${res.data.num_rows} rows from ${res.data.num_hospitals} hospitals</div>
      <table class="data-table"><thead><tr>${Object.keys(res.data.preview[0]||{}).map(k=>`<th>${k}</th>`).join('')}</tr></thead>
      <tbody>${res.data.preview.map(r=>`<tr>${Object.values(r).map(v=>`<td>${typeof v==='number'?v.toFixed?.(2)??v:v}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`;
  } catch (e) { toast(e.message, 'error'); }
  hideLoading();
}

// ── Privacy Budget Tab ──
async function loadPrivacyBudget() {
  if (!state.datasetId) { document.getElementById('privacy-content').innerHTML = '<div class="empty-state"><div class="empty-icon">🔐</div><p>Upload data and generate synthetic data to see privacy budget</p></div>'; return; }
  try {
    const res = await api.getPrivacyBudget(state.datasetId);
    const b = res.data;
    document.getElementById('privacy-content').innerHTML = `
      <div class="grid grid-2">
        <div class="card" style="text-align:center"><div class="card-title">🔐 Privacy Budget</div>
          <div class="gauge-container"><canvas id="privacy-gauge"></canvas><div class="gauge-label"><div class="gauge-value" style="color:${b.utilization_pct > 75 ? '#ef4444' : '#10b981'}">${b.utilization_pct}%</div><div class="gauge-subtitle">Budget Used</div></div></div></div>
        <div class="card"><div class="card-title">📊 Budget Details</div>
          <div class="grid grid-2">
            <div class="stat-card"><div class="stat-label">ε Used</div><div class="stat-value amber">${b.total_epsilon_used}</div></div>
            <div class="stat-card"><div class="stat-label">ε Remaining</div><div class="stat-value green">${b.remaining_epsilon}</div></div>
            <div class="stat-card"><div class="stat-label">Max Budget</div><div class="stat-value blue">${b.max_epsilon}</div></div>
            <div class="stat-card"><div class="stat-label">Queries</div><div class="stat-value purple">${b.num_queries}</div></div>
          </div>
          ${b.warning_level ? `<div class="badge badge-danger" style="margin-top:1rem">⚠ Warning: ${b.warning_level} utilization</div>` : ''}
        </div>
      </div>
      ${b.history.length ? `<div class="card"><div class="card-title">📜 Query History</div>
        <table class="data-table"><thead><tr><th>Operation</th><th>ε Spent</th><th>Cumulative ε</th></tr></thead>
        <tbody>${b.history.map(h => `<tr><td>${h.operation}</td><td>${h.epsilon}</td><td>${h.cumulative_epsilon?.toFixed(4) ?? '-'}</td></tr>`).join('')}</tbody></table></div>` : ''}`;
    setTimeout(() => {
      const color = b.utilization_pct > 75 ? 'rgba(239,68,68,1)' : b.utilization_pct > 50 ? 'rgba(245,158,11,1)' : 'rgba(16,185,129,1)';
      charts.gauge('privacy-gauge', b.utilization_pct, 100, color);
    }, 100);
  } catch (e) { console.error(e); }
}

// ── Model hint ──
function updateModelHint() {
  const model = document.getElementById('gen-model').value;
  const hint = document.getElementById('epoch-hint');
  if (model === 'statistical') {
    hint.textContent = '(ignored for Statistical)';
  } else {
    hint.textContent = model === 'ctgan' ? '(50 recommended for CTGAN)' : '(50 recommended for TVAE)';
  }
}

// ── Init ──
document.addEventListener('DOMContentLoaded', () => {
  // Tab navigation
  document.querySelectorAll('.nav-tab').forEach(tab => {
    tab.addEventListener('click', () => {
      switchTab(tab.dataset.tab);
      if (tab.dataset.tab === 'tab-privacy') loadPrivacyBudget();
    });
  });
  // File upload
  const uploadZone = document.getElementById('upload-zone');
  const fileInput = document.getElementById('file-input');
  if (uploadZone) {
    uploadZone.addEventListener('dragover', e => { e.preventDefault(); uploadZone.classList.add('dragover'); });
    uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('dragover'));
    uploadZone.addEventListener('drop', e => { e.preventDefault(); uploadZone.classList.remove('dragover'); if (e.dataTransfer.files[0]) handleFileUpload(e.dataTransfer.files[0]); });
  }
  if (fileInput) fileInput.addEventListener('change', e => { if (e.target.files[0]) handleFileUpload(e.target.files[0]); });
});
