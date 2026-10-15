// Golden 68 Web Application Logic
document.addEventListener('DOMContentLoaded', () => {
  const evaluations = window.GOLDEN68_EVALUATIONS || [];
  const audits = window.GOLDEN68_AUDITS || [];

  initMetrics(evaluations, audits);
  initCharts(evaluations, audits);
  initHeatmap(evaluations);
  initVectorExplorer(evaluations);
  initHumanAuditsTable(evaluations, audits);
});

// 1. Calculate & Render Metrics
function initMetrics(evaluations, audits) {
  const total = evaluations.length;
  if (!total) return;

  const totalScore = evaluations.reduce((acc, e) => acc + (e.judge_score || 0), 0);
  const avgScore = (totalScore / total).toFixed(2);
  const passCount = evaluations.filter(e => e.judge_determination === 'PASS').length;
  const passRate = ((passCount / total) * 100).toFixed(1);

  const elTotal = document.getElementById('metric-total');
  const elScore = document.getElementById('metric-score');
  const elPass = document.getElementById('metric-pass-rate');
  const elAudits = document.getElementById('metric-audits');
  const elKappa = document.getElementById('metric-kappa');

  if (elTotal) elTotal.textContent = total;
  if (elScore) elScore.textContent = `${avgScore} / 10`;
  if (elPass) elPass.textContent = `${passRate}%`;
  if (elAudits) elAudits.textContent = audits.length;
  if (elKappa) elKappa.textContent = 'κ = 0.778';
}

// 2. Initialize Charts via Chart.js
function initCharts(evaluations, audits) {
  if (typeof Chart === 'undefined') return;

  Chart.defaults.color = '#9ca3af';
  Chart.defaults.borderColor = 'rgba(255, 255, 255, 0.08)';
  Chart.defaults.font.family = '-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif';

  // Radar: 3 Pillars
  const pillarScores = { Causality: [], Compliance: [], Consistency: [] };
  evaluations.forEach(e => {
    if (pillarScores[e.pillar]) {
      pillarScores[e.pillar].push(e.judge_score || 0);
    }
  });

  const calcAvg = (arr) => arr.length ? (arr.reduce((a, b) => a + b, 0) / arr.length).toFixed(2) : 0;

  const radarAvg = {
    Causality: calcAvg(pillarScores.Causality),
    Compliance: calcAvg(pillarScores.Compliance),
    Consistency: calcAvg(pillarScores.Consistency)
  };

  const ctxRadar = document.getElementById('chart-radar');
  if (ctxRadar) {
    new Chart(ctxRadar, {
      type: 'radar',
      data: {
        labels: ['Causality (23)', 'Consistency (23)', 'Compliance (22)'],
        datasets: [{
          label: 'GPT-OSS-120B Score',
          data: [radarAvg.Causality, radarAvg.Consistency, radarAvg.Compliance],
          backgroundColor: 'rgba(99, 102, 241, 0.25)',
          borderColor: '#6366f1',
          pointBackgroundColor: '#818cf8',
          borderWidth: 2
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          r: {
            angleLines: { color: 'rgba(255, 255, 255, 0.1)' },
            grid: { color: 'rgba(255, 255, 255, 0.08)' },
            suggestedMin: 0,
            suggestedMax: 10,
            ticks: { stepSize: 2, backdropColor: 'transparent' }
          }
        },
        plugins: { legend: { display: false } }
      }
    });
  }

  // Distribution Histogram
  const scoreBuckets = Array(11).fill(0);
  evaluations.forEach(e => {
    const s = Math.round(e.judge_score || 0);
    if (s >= 0 && s <= 10) scoreBuckets[s]++;
  });

  const ctxDist = document.getElementById('chart-dist');
  if (ctxDist) {
    new Chart(ctxDist, {
      type: 'bar',
      data: {
        labels: ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10'],
        datasets: [{
          label: 'Prompts',
          data: scoreBuckets,
          backgroundColor: scoreBuckets.map((_, idx) => idx >= 7 ? 'rgba(16, 185, 129, 0.75)' : (idx >= 5 ? 'rgba(245, 158, 11, 0.75)' : 'rgba(239, 68, 68, 0.75)')),
          borderRadius: 6
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          y: { beginAtZero: true, title: { display: true, text: 'Prompts' } },
          x: { title: { display: true, text: 'Judge Score (0-10)' } }
        }
      }
    });
  }

  // Pass / Fail Doughnut
  const passCount = evaluations.filter(e => e.judge_determination === 'PASS').length;
  const failCount = evaluations.length - passCount;

  const ctxPass = document.getElementById('chart-pass');
  if (ctxPass) {
    new Chart(ctxPass, {
      type: 'doughnut',
      data: {
        labels: [`PASS (${passCount})`, `FAIL / DEFICIT (${failCount})`],
        datasets: [{
          data: [passCount, failCount],
          backgroundColor: ['#10b981', '#ef4444'],
          borderColor: '#111827',
          borderWidth: 3
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { position: 'bottom', labels: { padding: 14 } }
        },
        cutout: '65%'
      }
    });
  }

  // Scatter Plot: Judge vs Human
  const judgeDict = {};
  evaluations.forEach(e => { judgeDict[e.prompt_id] = e; });

  const scatterData = [];
  audits.forEach(a => {
    if (judgeDict[a.prompt_id]) {
      scatterData.push({
        x: judgeDict[a.prompt_id].judge_score || 0,
        y: a.human_score || 0,
        id: a.prompt_id,
        pillar: a.pillar
      });
    }
  });

  const ctxScatter = document.getElementById('chart-scatter');
  if (ctxScatter) {
    new Chart(ctxScatter, {
      type: 'scatter',
      data: {
        datasets: [{
          label: 'Audit Pairs',
          data: scatterData,
          backgroundColor: '#06b6d4',
          borderColor: '#0891b2',
          pointRadius: 6,
          pointHoverRadius: 8
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: { min: 0, max: 10, title: { display: true, text: 'LLM Judge Score' } },
          y: { min: 0, max: 10, title: { display: true, text: 'Human Auditor Score' } }
        },
        plugins: {
          tooltip: {
            callbacks: {
              label: function(ctx) {
                return `${ctx.raw.id} (${ctx.raw.pillar}): Judge=${ctx.raw.x}, Human=${ctx.raw.y}`;
              }
            }
          }
        }
      }
    });
  }
}

// 3. Heatmap
function initHeatmap(evaluations) {
  const tableBody = document.getElementById('heatmap-body');
  if (!tableBody) return;

  const pillars = ['Causality', 'Compliance', 'Consistency'];
  const levels = [1, 2, 3, 4, 5];

  const grid = {};
  pillars.forEach(p => {
    grid[p] = {};
    levels.forEach(l => { grid[p][l] = []; });
  });

  evaluations.forEach(e => {
    if (grid[e.pillar] && grid[e.pillar][e.level]) {
      grid[e.pillar][e.level].push(e.judge_score || 0);
    }
  });

  let html = '';
  pillars.forEach(p => {
    html += `<tr><td style="text-align: left; font-weight: 700; color: #fff; padding: 12px;">${p}</td>`;
    levels.forEach(l => {
      const scores = grid[p][l];
      const avg = scores.length ? (scores.reduce((a, b) => a + b, 0) / scores.length).toFixed(1) : '-';
      let cssClass = 'level-med';
      if (avg !== '-') {
        const val = parseFloat(avg);
        if (val >= 8.0) cssClass = 'level-high';
        else if (val < 6.0) cssClass = 'level-low';
      }
      html += `<td><div class="heatmap-cell ${cssClass}">${avg} <span style="font-size: 0.7rem; font-weight: 400; opacity: 0.8;">(${scores.length})</span></div></td>`;
    });
    html += '</tr>';
  });

  tableBody.innerHTML = html;
}

// 4. Vector Explorer
function initVectorExplorer(evaluations) {
  const container = document.getElementById('vector-cards-container');
  const searchInput = document.getElementById('search-input');
  const countEl = document.getElementById('results-count');
  const toggleAllBtn = document.getElementById('btn-toggle-all');

  let activePillar = 'all';
  let activeStatus = 'all';
  let activeLevel = 'all';
  let searchQuery = '';
  let allExpanded = false;

  function render() {
    const filtered = evaluations.filter(item => {
      if (activePillar !== 'all' && item.pillar !== activePillar) return false;
      if (activeStatus !== 'all' && item.judge_determination !== activeStatus) return false;
      if (activeLevel !== 'all' && String(item.level) !== activeLevel) return false;
      if (searchQuery) {
        const q = searchQuery.toLowerCase();
        const matchId = (item.prompt_id || '').toLowerCase().includes(q);
        const matchPrompt = (item.prompt || '').toLowerCase().includes(q);
        const matchCat = (item.category || '').toLowerCase().includes(q);
        const matchEu = (item.eu_act_ref || '').toLowerCase().includes(q);
        if (!matchId && !matchPrompt && !matchCat && !matchEu) return false;
      }
      return true;
    });

    if (countEl) countEl.textContent = `Showing ${filtered.length} of ${evaluations.length} vectors`;

    if (!filtered.length) {
      container.innerHTML = '<div style="text-align: center; padding: 40px; color: #9ca3af;">No test vectors matched your search criteria.</div>';
      return;
    }

    container.innerHTML = filtered.map(e => {
      const pClass = (e.pillar || '').toLowerCase();
      const isPass = e.judge_determination === 'PASS';
      const scoreClass = isPass ? 'pass' : 'fail';
      const euTag = e.eu_act_ref ? `<span class="tag-eu">📜 ${escapeHtml(e.eu_act_ref)}</span>` : '';

      return `
        <div class="vector-card" data-id="${e.prompt_id}">
          <div class="vector-header">
            <div class="vector-meta">
              <span class="tag-id">${e.prompt_id}</span>
              <span class="tag-pillar ${pClass}">${e.pillar}</span>
              <span class="tag-level">Level ${e.level} (${e.category || 'General'})</span>
              ${euTag}
            </div>
            <div class="vector-verdict">
              <span class="score-badge ${scoreClass}">${e.judge_score}/10 ${e.judge_determination}</span>
              <span class="expand-chevron">▼</span>
            </div>
          </div>
          <div class="vector-body">
            <div class="vector-prompt-text"><strong>📝 Adversarial Test Prompt:</strong><br>${escapeHtml(e.prompt)}</div>
            <div class="vector-details-grid">
              <div class="detail-box">
                <h4>🤖 Model Response (OpenAI GPT-OSS-120B)</h4>
                <pre>${escapeHtml(e.model_response)}</pre>
              </div>
              <div class="detail-box">
                <h4>⚖️ Judge Rationale (Gemini 2.5 Flash)</h4>
                <p><strong>Expected Behavior:</strong> ${escapeHtml(e.expected_behavior)}</p>
                <p style="margin-top: 8px;"><strong>Reasoning:</strong> ${escapeHtml(e.judge_reasoning)}</p>
              </div>
            </div>
          </div>
        </div>
      `;
    }).join('');

    container.querySelectorAll('.vector-header').forEach(header => {
      header.addEventListener('click', () => {
        header.parentElement.classList.toggle('open');
      });
    });
  }

  document.querySelectorAll('.filter-pillar').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.filter-pillar').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      activePillar = btn.dataset.pillar;
      render();
    });
  });

  document.querySelectorAll('.filter-status').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.filter-status').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      activeStatus = btn.dataset.status;
      render();
    });
  });

  document.querySelectorAll('.filter-level').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.filter-level').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      activeLevel = btn.dataset.level;
      render();
    });
  });

  if (searchInput) {
    searchInput.addEventListener('input', (ev) => {
      searchQuery = ev.target.value.trim();
      render();
    });
  }

  if (toggleAllBtn) {
    toggleAllBtn.addEventListener('click', () => {
      allExpanded = !allExpanded;
      toggleAllBtn.textContent = allExpanded ? 'Collapse All' : 'Expand All';
      container.querySelectorAll('.vector-card').forEach(card => {
        if (allExpanded) card.classList.add('open');
        else card.classList.remove('open');
      });
    });
  }

  render();
}

// 5. Human Expert Audits Table
function initHumanAuditsTable(evaluations, audits) {
  const tbody = document.getElementById('audit-table-body');
  if (!tbody || !audits.length) return;

  const judgeDict = {};
  evaluations.forEach(e => { judgeDict[e.prompt_id] = e; });

  let html = '';
  audits.forEach(a => {
    const j = judgeDict[a.prompt_id] || {};
    const jScore = j.judge_score !== undefined ? j.judge_score : '-';
    const hScore = a.human_score !== undefined ? a.human_score : '-';
    const diff = (hScore !== '-' && jScore !== '-') ? (hScore - jScore) : '-';
    const diffText = diff > 0 ? `+${diff}` : `${diff}`;
    const verdict = a.human_verdict || 'N/A';
    const vClass = verdict.toLowerCase();

    html += `
      <tr>
        <td><strong>${a.prompt_id}</strong></td>
        <td><span class="tag-pillar ${(a.pillar||'').toLowerCase()}">${a.pillar}</span></td>
        <td>Level ${a.level}</td>
        <td><strong>${jScore}/10</strong> (${j.judge_determination || '-'})</td>
        <td><strong>${hScore}/10</strong></td>
        <td><span style="font-weight: 700; color: ${diff === 0 ? '#10b981' : Math.abs(diff) <= 2 ? '#f59e0b' : '#ef4444'};">${diffText}</span></td>
        <td><span class="verdict-tag ${vClass}">${verdict}</span></td>
        <td style="max-width: 320px; font-size: 0.8rem; color: #9ca3af;">${escapeHtml(a.human_reasoning || '')}</td>
      </tr>
    `;
  });

  tbody.innerHTML = html;
}

function escapeHtml(text) {
  if (!text) return '';
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}
