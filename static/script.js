(() => {
  'use strict';
  const $ = (selector, root = document) => root.querySelector(selector);
  const state = {dataset: null, evaluation: null, models: [], ready: false};
  const numberFormat = new Intl.NumberFormat(undefined, {maximumFractionDigits: 2});
  const percent = value => value == null || !Number.isFinite(Number(value)) ? '—' : `${(Number(value) * 100).toFixed(2)}%`;
  const display = value => value == null ? '—' : (typeof value === 'number' ? numberFormat.format(value) : String(value));

  async function api(path, options = {}) {
    const response = await fetch(path, {headers: {Accept: 'application/json', ...(options.body ? {'Content-Type': 'application/json'} : {})}, ...options});
    let data;
    try { data = await response.json(); }
    catch { throw new Error(`The server returned an unreadable response for ${path}.`); }
    if (!response.ok) throw new Error(data.error || `${path} failed (${response.status}).`);
    if (!data || typeof data !== 'object') throw new Error(`The server returned malformed data for ${path}.`);
    return data;
  }

  function setHealth(status, healthy) {
    $('#healthText').textContent = status;
    $('#healthDot').style.background = healthy ? '#55b993' : '#b95c4a';
    $('#healthDot').style.boxShadow = healthy ? '0 0 0 3px #55b99322' : '0 0 0 3px #b95c4a22';
  }

  function showGlobalError(message) {
    const el = $('#globalError');
    el.textContent = message;
    el.hidden = !message;
  }

  function renderDataset(data) {
    if (!Array.isArray(data.columns) || !Array.isArray(data.features) || !Array.isArray(data.preview)) {
      throw new Error('Dataset response is missing columns, features, or preview records.');
    }
    state.dataset = data;
    $('#sourceName').textContent = data.dataset || 'Configured CSV';
    $('#statRows').textContent = display(data.rows);
    $('#statRowsNote').textContent = `${display(data.source_rows)} source rows · ${display(data.duplicate_rows_removed)} duplicates removed`;
    $('#statFeatures').textContent = display(data.feature_count);
    $('#targetName').textContent = data.target || 'Target';
    $('#footerTarget').textContent = `· Target: ${data.target || 'configured target'}`;
    $('#datasetNote').textContent = `${display(data.rows)} usable records from ${data.dataset || 'the configured CSV'}; date and identifier columns are excluded from model inputs.`;
    $('#cleanupNote').textContent = `Removed ${display(data.duplicate_rows_removed)} duplicate rows and ${display(data.invalid_target_rows_removed)} rows with unusable target values. Missing feature values are imputed inside each model pipeline.`;
    const distribution = data.target_distribution || {};
    const noPlacement = distribution.no_placement ?? 0;
    const placement = distribution.placement ?? 0;
    const noPlacementPct = distribution.no_placement_percent ?? 0;
    const placementPct = distribution.placement_percent ?? 0;
    $('#noPlacementCount').textContent = display(noPlacement);
    $('#placementCount').textContent = display(placement);
    $('#noPlacementPct').textContent = `${display(noPlacementPct)}%`;
    $('#placementPct').textContent = `${display(placementPct)}%`;
    $('#noPlacementBar').style.width = `${Math.max(0, Math.min(100, Number(noPlacementPct) || 0))}%`;
    $('#placementBar').style.width = `${Math.max(0, Math.min(100, Number(placementPct) || 0))}%`;
    $('#imbalanceAlert').hidden = !data.class_imbalance_warning;
    $('#imbalanceAlert').textContent = data.class_imbalance_warning
      ? 'The target classes are imbalanced. Accuracy can hide missed placement events; compare placement recall, F1-score, and ROC-AUC as well.'
      : '';
    renderColumns(data.columns);
    renderPreview(data.columns, data.preview);
    renderFeatures(data.features);
    $('#datasetStatus').textContent = `${data.columns.length} source columns inspected · ${data.missing_values} missing source values · target mapping ${JSON.stringify(data.target_mapping || {})}`;
  }

  function cell(row, value, className = '') {
    const td = document.createElement('td');
    if (className) td.className = className;
    td.textContent = value == null ? '—' : String(value);
    row.append(td);
    return td;
  }

  function renderColumns(columns) {
    const tbody = $('#columnsTable tbody');
    tbody.replaceChildren();
    columns.forEach(column => {
      const tr = document.createElement('tr');
      cell(tr, column.name, 'model-name');
      cell(tr, column.type);
      cell(tr, display(column.missing));
      cell(tr, display(column.unique));
      const usage = column.is_target ? 'Target' : column.is_feature ? 'Input' : 'Excluded';
      const td = cell(tr, usage);
      const tag = document.createElement('span'); tag.className = 'tag'; tag.textContent = usage; td.replaceChildren(tag);
      tbody.append(tr);
    });
  }

  function renderPreview(columns, records) {
    const table = $('#previewTable');
    const thead = document.createElement('thead');
    const headRow = document.createElement('tr');
    columns.forEach(column => { const th = document.createElement('th'); th.scope = 'col'; th.textContent = column.name; headRow.append(th); });
    thead.append(headRow);
    const tbody = document.createElement('tbody');
    records.forEach(record => {
      const tr = document.createElement('tr');
      columns.forEach(column => cell(tr, record[column.name]));
      tbody.append(tr);
    });
    table.replaceChildren(thead, tbody);
    const caption = document.createElement('caption'); caption.textContent = 'Preview of records from the configured dataset'; table.prepend(caption);
  }

  function renderFeatures(features) {
    const container = $('#featureFields'); container.replaceChildren();
    features.forEach((feature, index) => {
      const wrap = document.createElement('div'); wrap.className = 'feature-field';
      const label = document.createElement('label');
      const id = `feature-${index}`; label.htmlFor = id; label.textContent = feature.name;
      let input;
      if (feature.type === 'category' && Array.isArray(feature.options) && feature.options.length) {
        input = document.createElement('select');
        feature.options.forEach(option => { const item = document.createElement('option'); item.value = option; item.textContent = option; input.append(item); });
        input.value = feature.default ?? feature.options[0];
      } else {
        input = document.createElement('input');
        input.type = feature.type === 'number' ? 'number' : 'text';
        if (input.type === 'number') { input.step = feature.step || 'any'; input.value = feature.default ?? ''; }
        else { input.value = feature.default ?? ''; }
      }
      input.id = id; input.name = feature.name; input.required = true;
      if (feature.type === 'number') input.inputMode = 'decimal';
      label.append(input); wrap.append(label);
      const hint = document.createElement('small'); hint.className = 'feature-hint';
      hint.textContent = feature.type === 'number' ? `Observed range: ${display(feature.minimum)} – ${display(feature.maximum)}` : `${feature.options?.length ? `${feature.options.length} observed categories` : 'Enter a category value'}`;
      wrap.append(hint); container.append(wrap);
    });
    $('#predictionStatus').textContent = `${features.length} model input fields loaded from the dataset schema.`;
  }

  function modelMetrics(model) { return model.test || model; }

  function renderEvaluation(data) {
    if (!Array.isArray(data.models)) throw new Error('Metrics response does not contain a model list.');
    state.evaluation = data; state.models = data.models.filter(model => model.status !== 'error');
    $('#statModels').textContent = String(state.models.length);
    $('#statModelNote').textContent = data.model_errors?.length ? `${data.model_errors.length} model(s) reported an error` : 'All available model runs reported';
    $('#bestModelHeading').textContent = data.best_model || '—';
    $('#selectionMetric').textContent = data.selection_metric || 'Selected using validation results';
    const bestModel = state.models.find(model => model.name === data.best_model);
    $('#statBestF1').textContent = bestModel ? percent(bestModel.validation_f1) : '—';
    $('#statBestName').textContent = bestModel?.name || 'No selected model';
    $('#splitInfo').textContent = `${data.split || 'Holdout evaluation'}  ·  train ${display(data.train_rows)}  ·  validation ${display(data.validation_rows)}  ·  test ${display(data.test_rows)}`;
    renderModelTable(state.models);
    renderModelSelectors(state.models, data.best_model);
    renderChart();
    renderConfusion();
    renderModelErrors(data.model_errors || []);
    $('#metricsStatus').textContent = `${state.models.length} model(s) evaluated. Best model selected by validation F1; reported model metrics use the held-out test partition.`;
    state.ready = state.models.length > 0;
    $('#predictButton').disabled = !state.ready;
    if (state.ready) $('#predictionStatus').textContent = 'A trained model is ready for inference.';
  }

  function renderModelTable(models) {
    const tbody = $('#modelTable tbody'); tbody.replaceChildren();
    if (!models.length) { const tr = document.createElement('tr'); cell(tr, 'No models are available.', ''); tr.firstChild.colSpan = 6; tbody.append(tr); return; }
    models.forEach(model => {
      const metrics = modelMetrics(model); const tr = document.createElement('tr');
      const nameCell = cell(tr, model.name, 'model-name');
      const sub = document.createElement('small'); sub.textContent = `validation F1 ${percent(model.validation_f1)}`; nameCell.append(sub);
      if (model.best) { const badge = document.createElement('span'); badge.className = 'best-pill'; badge.textContent = 'BEST'; nameCell.append(badge); }
      ['accuracy', 'precision', 'recall', 'f1', 'roc_auc'].forEach(key => cell(tr, percent(metrics[key]), 'metric-value'));
      tbody.append(tr);
    });
  }

  function renderModelSelectors(models, bestName) {
    for (const selector of ['#matrixModel', '#predictionModel']) {
      const select = $(selector); const old = select.value; select.replaceChildren();
      models.forEach(model => { const option = document.createElement('option'); option.value = model.name; option.textContent = model.name; select.append(option); });
      select.value = models.some(model => model.name === old) ? old : (models.some(model => model.name === bestName) ? bestName : models[0]?.name || '');
    }
  }

  function renderChart() {
    const container = $('#metricChart'); container.replaceChildren();
    const metric = $('#chartMetric').value;
    const values = state.models.map(model => ({name: model.name, value: Number(modelMetrics(model)[metric]), best: Boolean(model.best)})).filter(item => Number.isFinite(item.value));
    container.setAttribute('aria-label', `${metric} comparison for ${values.length} evaluated models`);
    if (!values.length) { const text = document.createElement('p'); text.className = 'muted-copy'; text.textContent = 'No values are available for this metric.'; container.append(text); return; }
    values.forEach(item => {
      const row = document.createElement('div'); row.className = 'chart-row';
      const name = document.createElement('span'); name.className = 'chart-name'; name.textContent = item.name;
      const track = document.createElement('span'); track.className = 'chart-track'; track.setAttribute('aria-hidden', 'true');
      const fill = document.createElement('span'); fill.className = `chart-fill${item.best ? ' best' : ''}`; fill.style.display = 'block'; fill.style.width = `${Math.max(0, Math.min(100, item.value * 100))}%`; track.append(fill);
      const score = document.createElement('span'); score.className = 'chart-score'; score.textContent = percent(item.value);
      row.append(name, track, score); container.append(row);
    });
  }

  function renderConfusion() {
    const container = $('#confusionMatrix'); container.replaceChildren();
    const model = state.models.find(item => item.name === $('#matrixModel').value);
    const matrix = model && modelMetrics(model).confusion_matrix;
    if (!Array.isArray(matrix) || matrix.length !== 2) { const p = document.createElement('p'); p.className = 'muted-copy'; p.textContent = 'Confusion matrix not available for this model.'; container.append(p); return; }
    const cells = [
      ['Actual / predicted', 'No placement', 'Placement'],
      ['No placement', String(matrix[0]?.[0] ?? '—'), String(matrix[0]?.[1] ?? '—')],
      ['Placement', String(matrix[1]?.[0] ?? '—'), String(matrix[1]?.[1] ?? '—')],
    ];
    cells.forEach((row, rowIndex) => row.forEach((value, colIndex) => {
      const item = document.createElement('div');
      if (rowIndex === 0 || colIndex === 0) { item.className = 'matrix-axis'; item.textContent = value; }
      else { item.className = `matrix-cell${rowIndex !== colIndex ? ' error' : ''}`; const strong = document.createElement('strong'); strong.textContent = value; const small = document.createElement('small'); small.textContent = rowIndex === colIndex ? 'correct' : 'misclassified'; item.append(strong, small); }
      container.append(item);
    }));
  }

  function renderModelErrors(errors) {
    const container = $('#modelErrors'); container.replaceChildren(); container.hidden = !errors.length;
    if (!errors.length) return;
    errors.forEach(error => { const line = document.createElement('div'); line.textContent = `${error.name}: ${error.error}`; container.append(line); });
  }

  async function loadAll() {
    showGlobalError(''); setHealth('CONNECTING', false);
    $('#metricsStatus').textContent = 'Loading model metrics; first-time training may take a few minutes.';
    const failures = [];
    const tasks = [
      api('/api/health').then(data => setHealth(data.status === 'ok' ? 'API CONNECTED' : 'API ISSUE', data.status === 'ok'))
        .catch(error => { setHealth('API OFFLINE', false); failures.push(error.message); }),
      api('/api/dataset').then(data => {
        try { renderDataset(data); $('#datasetStatus').textContent += ' · Loaded successfully.'; }
        catch (error) { failures.push(error.message); }
      }).catch(error => { $('#datasetStatus').textContent = `Dataset unavailable: ${error.message}`; failures.push(error.message); }),
      api('/api/metrics').then(data => {
        try { renderEvaluation(data); }
        catch (error) { failures.push(error.message); }
      }).catch(error => { $('#metricsStatus').textContent = `Model metrics unavailable: ${error.message}`; failures.push(error.message); }),
    ];
    await Promise.all(tasks);
    if (failures.length) showGlobalError(failures.join(' '));
  }

  function setBusy(button, busy, busyText, idleText) { button.disabled = busy; button.textContent = busy ? busyText : idleText; }

  $('#chartMetric').addEventListener('change', renderChart);
  $('#matrixModel').addEventListener('change', renderConfusion);
  $('#predictionModel').addEventListener('change', () => { $('#predictionResult').hidden = true; });

  $('#predictionForm').addEventListener('submit', async event => {
    event.preventDefault();
    if (!state.ready) return;
    const button = $('#predictButton'); setBusy(button, true, 'Predicting…', 'Estimate placement');
    const status = $('#predictionStatus'); status.textContent = 'Applying the trained pipeline…';
    const payload = Object.fromEntries(new FormData(event.currentTarget).entries());
    try {
      const data = await api('/api/predict', {method: 'POST', body: JSON.stringify(payload)});
      const result = $('#predictionResult'); result.replaceChildren(); result.hidden = false;
      const label = document.createElement('div');
      const heading = document.createElement('strong'); heading.textContent = data.placement_label || data.label || `Class ${data.prediction}`;
      const detail = document.createElement('p'); detail.textContent = `${data.model} · ${data.target} · predicted class ${data.prediction}`;
      label.append(heading, detail); result.append(label);
      const probabilityValue = Number(data.placement_probability);
      if (data.probability_supported && Number.isFinite(probabilityValue)) {
        const probability = document.createElement('span'); probability.className = 'probability'; probability.textContent = `${(probabilityValue * 100).toFixed(1)}% placement probability`; result.append(probability);
      } else {
        const probability = document.createElement('span'); probability.className = 'muted-copy'; probability.textContent = 'Probability not provided by this model'; result.append(probability);
      }
      result.classList.toggle('placement', Number(data.prediction) === 1); result.classList.remove('error'); status.textContent = 'Prediction completed successfully.';
    } catch (error) {
      const result = $('#predictionResult'); result.replaceChildren(); result.hidden = false; result.className = 'prediction-result error';
      const message = document.createElement('strong'); message.textContent = error.message; result.append(message); status.textContent = 'Prediction failed. Correct the input and try again.';
    } finally { setBusy(button, false, '', 'Estimate placement'); button.disabled = !state.ready; }
  });

  $('#retrainModels').addEventListener('click', async () => {
    const button = $('#retrainModels'); setBusy(button, true, 'Training models…', 'Retrain models'); showGlobalError('');
    $('#metricsStatus').textContent = 'Training and evaluating the configured models. This can take a few minutes…';
    try { const data = await api('/api/retrain', {method: 'POST', body: JSON.stringify({})}); renderEvaluation(data); $('#metricsStatus').textContent = 'Retraining completed; metrics were recalculated on the held-out partitions.'; }
    catch (error) { showGlobalError(error.message); $('#metricsStatus').textContent = 'Retraining failed.'; }
    finally { setBusy(button, false, '', 'Retrain models'); }
  });

  $('#refreshDataset').addEventListener('click', async () => {
    const button = $('#refreshDataset'); setBusy(button, true, 'Reloading and training…', 'Reload dataset & retrain'); showGlobalError('');
    $('#datasetStatus').textContent = 'Reloading the configured CSV and retraining models…';
    try { await api('/api/dataset/reload', {method: 'POST', body: JSON.stringify({})}); await loadAll(); $('#datasetStatus').textContent += ' · Dataset reload and retraining completed.'; }
    catch (error) { showGlobalError(error.message); $('#datasetStatus').textContent = `Dataset reload failed: ${error.message}`; }
    finally { setBusy(button, false, '', 'Reload dataset & retrain'); }
  });

  const navLinks = [...document.querySelectorAll('.nav-link')];
  function syncNavigation() {
    const activeHash = location.hash || '#overview';
    navLinks.forEach(link => { const active = link.hash === activeHash; link.classList.toggle('active', active); if (active) link.setAttribute('aria-current', 'page'); else link.removeAttribute('aria-current'); });
  }
  navLinks.forEach(link => link.addEventListener('click', () => { if (location.hash === link.hash) syncNavigation(); }));
  window.addEventListener('hashchange', syncNavigation);
  syncNavigation();
  loadAll();
})();
