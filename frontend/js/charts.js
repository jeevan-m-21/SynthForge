/**
 * MediSynth.AI — Chart Utilities (Chart.js)
 */
const ChartColors = {
  blue: 'rgba(59,130,246,1)', blueBg: 'rgba(59,130,246,0.15)',
  purple: 'rgba(139,92,246,1)', purpleBg: 'rgba(139,92,246,0.15)',
  cyan: 'rgba(6,182,212,1)', cyanBg: 'rgba(6,182,212,0.15)',
  green: 'rgba(16,185,129,1)', greenBg: 'rgba(16,185,129,0.15)',
  amber: 'rgba(245,158,11,1)', amberBg: 'rgba(245,158,11,0.15)',
  red: 'rgba(239,68,68,1)', redBg: 'rgba(239,68,68,0.15)',
  pink: 'rgba(236,72,153,1)',
  textMuted: 'rgba(148,163,184,0.6)',
  gridLine: 'rgba(255,255,255,0.06)',
};

function getChartOpts() {
  const isLight = document.documentElement.getAttribute('data-theme') === 'light';
  const tickColor = isLight ? '#64748b' : '#94a3b8';
  const gridColor = isLight ? 'rgba(0,0,0,0.06)' : 'rgba(255,255,255,0.06)';
  const labelColor = isLight ? '#334155' : '#94a3b8';

  return {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        labels: {
          color: labelColor,
          font: { family: 'Inter', size: 11, weight: '500' },
        },
      },
    },
    scales: {
      x: { ticks: { color: tickColor, font: { size: 10 } }, grid: { color: gridColor } },
      y: { ticks: { color: tickColor, font: { size: 10 } }, grid: { color: gridColor } },
    },
  };
}


const charts = {
  _instances: {},

  destroy(id) {
    if (this._instances[id]) { this._instances[id].destroy(); delete this._instances[id]; }
  },

  _create(id, cfg) {
    this.destroy(id);
    const ctx = document.getElementById(id);
    if (!ctx) return null;
    this._instances[id] = new Chart(ctx, cfg);
    return this._instances[id];
  },

  distribution(canvasId, data, colName) {
    const opts = getChartOpts();
    const isLight = document.documentElement.getAttribute('data-theme') === 'light';
    return this._create(canvasId, {
      type: 'bar',
      data: {
        labels: data.bins.map(b => b.toFixed(1)),
        datasets: [
          { label: 'Real', data: data.real_counts, backgroundColor: ChartColors.blueBg, borderColor: ChartColors.blue, borderWidth: 1 },
          { label: 'Synthetic', data: data.synth_counts, backgroundColor: ChartColors.purpleBg, borderColor: ChartColors.purple, borderWidth: 1 },
        ],
      },
      options: {
        ...opts,
        plugins: {
          ...opts.plugins,
          title: { display: true, text: colName, color: isLight ? '#1e293b' : '#f1f5f9', font: { size: 13, weight: '600' } },
        },
      },
    });
  },

  categoricalDist(canvasId, data, colName) {
    const opts = getChartOpts();
    const isLight = document.documentElement.getAttribute('data-theme') === 'light';
    return this._create(canvasId, {
      type: 'bar',
      data: {
        labels: data.categories,
        datasets: [
          { label: 'Real', data: data.real_proportions, backgroundColor: ChartColors.blue },
          { label: 'Synthetic', data: data.synth_proportions, backgroundColor: ChartColors.purple },
        ],
      },
      options: {
        ...opts,
        plugins: {
          ...opts.plugins,
          title: { display: true, text: colName, color: isLight ? '#1e293b' : '#f1f5f9', font: { size: 13, weight: '600' } },
        },
      },
    });
  },

  comparisonBar(canvasId, labels, realScores, synthScores) {
    const opts = getChartOpts();
    return this._create(canvasId, {
      type: 'bar',
      data: {
        labels,
        datasets: [
          { label: 'Train on Real', data: realScores, backgroundColor: ChartColors.blue, borderRadius: 4 },
          { label: 'Train on Synthetic', data: synthScores, backgroundColor: ChartColors.cyan, borderRadius: 4 },
        ],
      },
      options: {
        ...opts,
        scales: {
          ...opts.scales,
          y: { ...opts.scales.y, min: 0, max: 1 },
        },
      },
    });
  },

  rocCurve(canvasId, fpr, tpr, label) {
    const opts = getChartOpts();
    const isLight = document.documentElement.getAttribute('data-theme') === 'light';
    const axisTitleColor = isLight ? '#475569' : '#94a3b8';
    return this._create(canvasId, {
      type: 'line',
      data: {
        labels: fpr,
        datasets: [
          { label: label || 'ROC', data: tpr, borderColor: ChartColors.cyan, backgroundColor: ChartColors.cyanBg, fill: true, tension: 0.3, pointRadius: 0 },
          { label: 'Random', data: fpr, borderColor: ChartColors.textMuted, borderDash: [5, 5], pointRadius: 0 },
        ],
      },
      options: {
        ...opts,
        scales: {
          x: { ...opts.scales.x, title: { display: true, text: 'FPR', color: axisTitleColor } },
          y: { ...opts.scales.y, title: { display: true, text: 'TPR', color: axisTitleColor } },
        },
      },
    });
  },

  radar(canvasId, labels, values) {
    const isLight = document.documentElement.getAttribute('data-theme') === 'light';
    const tickColor = isLight ? '#64748b' : '#94a3b8';
    const gridColor = isLight ? 'rgba(0,0,0,0.08)' : 'rgba(255,255,255,0.06)';
    const labelColor = isLight ? '#334155' : '#94a3b8';
    return this._create(canvasId, {
      type: 'radar',
      data: {
        labels,
        datasets: [{
          label: 'Risk Score', data: values,
          backgroundColor: 'rgba(239,68,68,0.15)', borderColor: ChartColors.red, pointBackgroundColor: ChartColors.red,
        }],
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        scales: {
          r: {
            beginAtZero: true, max: 100,
            ticks: { color: tickColor, backdropColor: 'transparent' },
            grid: { color: gridColor },
            pointLabels: { color: labelColor, font: { size: 11, weight: '500' } },
          },
        },
        plugins: { legend: { display: false } },
      },
    });
  },

  gauge(canvasId, value, max, color) {
    return this._create(canvasId, {
      type: 'doughnut',
      data: {
        datasets: [{
          data: [value, max - value],
          backgroundColor: [color || ChartColors.blue, 'rgba(255,255,255,0.05)'],
          borderWidth: 0, cutout: '80%', circumference: 270, rotation: 225,
        }],
      },
      options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false }, tooltip: { enabled: false } } },
    });
  },

  heatmapHTML(container, matrix, labels) {
    if (!matrix || !matrix.length) return;
    const getColor = v => {
      const abs = Math.abs(v);
      if (abs > 0.7) return v > 0 ? 'rgba(59,130,246,0.7)' : 'rgba(239,68,68,0.7)';
      if (abs > 0.4) return v > 0 ? 'rgba(59,130,246,0.4)' : 'rgba(239,68,68,0.4)';
      if (abs > 0.2) return v > 0 ? 'rgba(59,130,246,0.2)' : 'rgba(239,68,68,0.2)';
      return 'rgba(255,255,255,0.05)';
    };
    let html = '<table class="heatmap-table"><tr><th></th>';
    labels.forEach(l => html += `<th>${l.substring(0, 8)}</th>`);
    html += '</tr>';
    matrix.forEach((row, i) => {
      html += `<tr><th>${labels[i].substring(0, 8)}</th>`;
      row.forEach(v => html += `<td style="background:${getColor(v)}">${v.toFixed(2)}</td>`);
      html += '</tr>';
    });
    html += '</table>';
    document.getElementById(container).innerHTML = html;
  },
};
