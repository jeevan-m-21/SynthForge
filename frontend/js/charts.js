/**
 * SynthForge — Chart Utilities (Chart.js)
 */
const ChartColors = {
  blue: 'rgba(59,130,246,1)', blueBg: 'rgba(59,130,246,0.18)',
  purple: 'rgba(139,92,246,1)', purpleBg: 'rgba(139,92,246,0.18)',
  cyan: 'rgba(6,182,212,1)', cyanBg: 'rgba(6,182,212,0.18)',
  green: 'rgba(16,185,129,1)', greenBg: 'rgba(16,185,129,0.18)',
  amber: 'rgba(245,158,11,1)', amberBg: 'rgba(245,158,11,0.18)',
  red: 'rgba(239,68,68,1)', redBg: 'rgba(239,68,68,0.18)',
  pink: 'rgba(236,72,153,1)',
  textMuted: 'rgba(148,163,184,0.7)',
};

function getChartOpts() {
  const isLight = document.documentElement.getAttribute('data-theme') === 'light';
  const tickColor = isLight ? '#475569' : '#94a3b8';
  const gridColor = isLight ? 'rgba(0,0,0,0.07)' : 'rgba(255,255,255,0.06)';
  const labelColor = isLight ? '#1e293b' : '#f1f5f9';

  return {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        labels: {
          color: labelColor,
          font: { family: 'Inter', size: 12, weight: '500' },
          boxWidth: 14,
          padding: 12,
        },
      },
      tooltip: {
        backgroundColor: isLight ? 'rgba(255,255,255,0.95)' : 'rgba(17,24,39,0.95)',
        titleColor: isLight ? '#0f172a' : '#f8fafc',
        bodyColor: isLight ? '#334155' : '#e2e8f0',
        borderColor: isLight ? 'rgba(0,0,0,0.1)' : 'rgba(255,255,255,0.15)',
        borderWidth: 1,
        padding: 10,
        cornerRadius: 8,
      },
    },
    scales: {
      x: {
        ticks: { color: tickColor, font: { size: 11 } },
        grid: { color: gridColor },
      },
      y: {
        ticks: { color: tickColor, font: { size: 11 } },
        grid: { color: gridColor },
      },
    },
  };
}

const charts = {
  _instances: {},

  destroy(id) {
    if (this._instances[id]) {
      try { this._instances[id].destroy(); } catch (e) {}
      delete this._instances[id];
    }
  },

  destroyAll() {
    Object.keys(this._instances).forEach(id => {
      try { this._instances[id].destroy(); } catch (e) {}
      delete this._instances[id];
    });
    this._instances = {};
  },

  resizeAll() {
    Object.values(this._instances).forEach(chart => {
      try { chart.resize(); } catch (e) {}
    });
  },

  _create(id, cfg) {
    this.destroy(id);
    const ctx = document.getElementById(id);
    if (!ctx) return null;
    try {
      this._instances[id] = new Chart(ctx, cfg);
      return this._instances[id];
    } catch (e) {
      console.warn(`Chart initialization error on [${id}]:`, e);
      return null;
    }
  },

  distribution(canvasId, data, colName) {
    const opts = getChartOpts();
    const isLight = document.documentElement.getAttribute('data-theme') === 'light';
    const labels = (data.bins || []).map(b => (typeof b === 'number' ? b.toFixed(1) : String(b)));

    return this._create(canvasId, {
      type: 'bar',
      data: {
        labels,
        datasets: [
          {
            label: 'Real Baseline',
            data: data.real_counts || [],
            backgroundColor: ChartColors.blueBg,
            borderColor: ChartColors.blue,
            borderWidth: 1.5,
            borderRadius: 3,
          },
          {
            label: 'Synthetic Generated',
            data: data.synth_counts || [],
            backgroundColor: ChartColors.purpleBg,
            borderColor: ChartColors.purple,
            borderWidth: 1.5,
            borderRadius: 3,
          },
        ],
      },
      options: {
        ...opts,
        plugins: {
          ...opts.plugins,
          title: {
            display: true,
            text: `Distribution: ${colName}`,
            color: isLight ? '#0f172a' : '#f8fafc',
            font: { size: 13, weight: '600' },
            padding: { bottom: 12 },
          },
        },
      },
    });
  },

  categoricalDist(canvasId, data, colName) {
    const opts = getChartOpts();
    const isLight = document.documentElement.getAttribute('data-theme') === 'light';
    const categories = data.categories || [];

    return this._create(canvasId, {
      type: 'bar',
      data: {
        labels: categories,
        datasets: [
          {
            label: 'Real Proportions',
            data: (data.real_proportions || []).map(v => (v != null ? +(v * 100).toFixed(1) : 0)),
            backgroundColor: ChartColors.blue,
            borderRadius: 4,
          },
          {
            label: 'Synthetic Proportions',
            data: (data.synth_proportions || []).map(v => (v != null ? +(v * 100).toFixed(1) : 0)),
            backgroundColor: ChartColors.purple,
            borderRadius: 4,
          },
        ],
      },
      options: {
        ...opts,
        plugins: {
          ...opts.plugins,
          title: {
            display: true,
            text: `Category Frequency (%): ${colName}`,
            color: isLight ? '#0f172a' : '#f8fafc',
            font: { size: 13, weight: '600' },
            padding: { bottom: 12 },
          },
          tooltip: {
            ...opts.plugins.tooltip,
            callbacks: {
              label: (ctx) => `${ctx.dataset.label}: ${ctx.raw}%`,
            },
          },
        },
        scales: {
          ...opts.scales,
          y: {
            ...opts.scales.y,
            title: { display: true, text: 'Percentage (%)', color: isLight ? '#64748b' : '#94a3b8' },
            ticks: { callback: v => `${v}%` },
          },
        },
      },
    });
  },

  comparisonBar(canvasId, labels, realScores, synthScores) {
    const opts = getChartOpts();
    const isLight = document.documentElement.getAttribute('data-theme') === 'light';
    return this._create(canvasId, {
      type: 'bar',
      data: {
        labels,
        datasets: [
          {
            label: 'Train on Real (TRTR Baseline)',
            data: realScores,
            backgroundColor: isLight ? 'rgba(37,99,235,0.85)' : 'rgba(59,130,246,0.85)',
            borderRadius: 4,
          },
          {
            label: 'Train on Synthetic (TSTR)',
            data: synthScores,
            backgroundColor: isLight ? 'rgba(8,145,178,0.85)' : 'rgba(6,182,212,0.85)',
            borderRadius: 4,
          },
        ],
      },
      options: {
        ...opts,
        scales: {
          ...opts.scales,
          y: {
            ...opts.scales.y,
            min: 0,
            max: 1.0,
            ticks: { callback: v => `${Math.round(v * 100)}%` },
          },
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
        labels: fpr || [0, 1],
        datasets: [
          {
            label: label || 'ROC Model Curve',
            data: tpr || [0, 1],
            borderColor: ChartColors.cyan,
            backgroundColor: ChartColors.cyanBg,
            fill: true,
            tension: 0.25,
            pointRadius: 0,
            borderWidth: 2,
          },
          {
            label: 'Random Chance Baseline',
            data: fpr || [0, 1],
            borderColor: ChartColors.textMuted,
            borderDash: [5, 5],
            pointRadius: 0,
            borderWidth: 1.5,
          },
        ],
      },
      options: {
        ...opts,
        scales: {
          x: {
            ...opts.scales.x,
            title: { display: true, text: 'False Positive Rate (FPR)', color: axisTitleColor },
            min: 0, max: 1,
          },
          y: {
            ...opts.scales.y,
            title: { display: true, text: 'True Positive Rate (TPR)', color: axisTitleColor },
            min: 0, max: 1,
          },
        },
      },
    });
  },

  radar(canvasId, labels, values) {
    const isLight = document.documentElement.getAttribute('data-theme') === 'light';
    const tickColor = isLight ? '#64748b' : '#94a3b8';
    const gridColor = isLight ? 'rgba(0,0,0,0.1)' : 'rgba(255,255,255,0.08)';
    const labelColor = isLight ? '#1e293b' : '#f1f5f9';

    return this._create(canvasId, {
      type: 'radar',
      data: {
        labels: labels || ['MIA', 'Re-ID', 'Attribute Inference'],
        datasets: [{
          label: 'Empirical Risk Score',
          data: values || [0, 0, 0],
          backgroundColor: 'rgba(239,68,68,0.2)',
          borderColor: ChartColors.red,
          pointBackgroundColor: ChartColors.red,
          pointBorderColor: '#ffffff',
          pointHoverBackgroundColor: '#ffffff',
          pointHoverBorderColor: ChartColors.red,
          borderWidth: 2,
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          r: {
            beginAtZero: true,
            min: 0,
            max: 100,
            ticks: {
              color: tickColor,
              backdropColor: 'transparent',
              stepSize: 25,
            },
            grid: { color: gridColor },
            angleLines: { color: gridColor },
            pointLabels: {
              color: labelColor,
              font: { family: 'Inter', size: 12, weight: '600' },
              padding: 8,
            },
          },
        },
        plugins: {
          legend: { display: false },
          tooltip: {
            ...getChartOpts().plugins.tooltip,
            callbacks: {
              label: (ctx) => `Risk: ${ctx.raw}/100`,
            },
          },
        },
      },
    });
  },

  heatmapHTML(containerId, matrix, labels) {
    const el = document.getElementById(containerId);
    if (!el) return;
    if (!matrix || !matrix.length || !labels || !labels.length) {
      el.innerHTML = '<p style="color:var(--text-muted);padding:1rem">No correlation data available.</p>';
      return;
    }

    const getColor = (v) => {
      const abs = Math.abs(v);
      if (abs > 0.7) return v > 0 ? 'rgba(59,130,246,0.75)' : 'rgba(239,68,68,0.75)';
      if (abs > 0.4) return v > 0 ? 'rgba(59,130,246,0.45)' : 'rgba(239,68,68,0.45)';
      if (abs > 0.15) return v > 0 ? 'rgba(59,130,246,0.22)' : 'rgba(239,68,68,0.22)';
      return 'rgba(148,163,184,0.06)';
    };

    let html = `
      <div class="heatmap-wrapper">
        <table class="heatmap-table">
          <thead>
            <tr>
              <th class="heatmap-corner"></th>
              ${labels.map(l => `<th class="heatmap-col-header" title="${l}"><div class="heatmap-header-text">${l}</div></th>`).join('')}
            </tr>
          </thead>
          <tbody>
    `;

    matrix.forEach((row, i) => {
      const rowLabel = labels[i] || `Col ${i}`;
      html += `<tr><th class="heatmap-row-header" title="${rowLabel}"><div class="heatmap-header-text">${rowLabel}</div></th>`;
      row.forEach((v, j) => {
        const val = typeof v === 'number' ? v : 0;
        const colLabel = labels[j] || `Col ${j}`;
        const titleText = `${rowLabel} ↔ ${colLabel}: ${val.toFixed(3)}`;
        html += `<td style="background:${getColor(val)}" title="${titleText}" class="heatmap-cell">${val.toFixed(2)}</td>`;
      });
      html += '</tr>';
    });

    html += `
          </tbody>
        </table>
      </div>
      <div class="heatmap-legend">
        <span class="legend-item"><span class="legend-box neg"></span> Negative (-1.0)</span>
        <span class="legend-item"><span class="legend-box neutral"></span> Weak / None (0.0)</span>
        <span class="legend-item"><span class="legend-box pos"></span> Positive (+1.0)</span>
      </div>
    `;

    el.innerHTML = html;
  },
};

