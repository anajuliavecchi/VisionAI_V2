
"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import {
  Activity,
  AlertTriangle,
  BarChart3,
  Camera,
  CheckCircle2,
  ChevronRight,
  CircleHelp,
  Cpu,
  Download,
  LayoutDashboard,
  LoaderCircle,
  Plus,
  RefreshCw,
  Search,
  Settings,
  Trash2,
  ShieldCheck,
  Users,
  Wifi,
  WifiOff,
  XCircle,
} from "lucide-react";

type Face = {
  name: string;
  authorized: boolean;
  score: number;
  margin?: number;
  bbox?: number[];
};

type Status = {
  status: string;
  environment: string;
  people_count: number;
  authorized_count: number;
  unauthorized_count: number;
  faces: Face[];
  camera_online: boolean;
  ai_online: boolean;
  fps: number;
  inference_ms: number;
};

type Enrollment = {
  active: boolean;
  name: string | null;
  samples: number;
  required: number;
  maximum: number;
  message: string;
};

type EventRecord = {
  id: number;
  event_type: string;
  severity: string;
  title: string;
  description: string;
  first_seen_at: string;
  confirmed_at: string;
  resolved_at: string | null;
  duration_seconds: number | null;
  people_count: number;
  status: "active" | "resolved";
};

type EventSummary = {
  total: number;
  active: number;
  resolved: number;
  today: number;
};

type Tab =
  | "overview"
  | "camera"
  | "people"
  | "alerts"
  | "reports"
  | "settings";

const API_URL = (
  process.env.NEXT_PUBLIC_VISIONAI_API_URL ||
  "http://192.168.50.20:8000"
).replace(/\/$/, "");

const initialStatus: Status = {
  status: "initializing",
  environment: "Conectando ao VisionAI...",
  people_count: 0,
  authorized_count: 0,
  unauthorized_count: 0,
  faces: [],
  camera_online: false,
  ai_online: false,
  fps: 0,
  inference_ms: 0,
};

const nav: {
  id: Tab;
  label: string;
  icon: typeof LayoutDashboard;
}[] = [
  { id: "overview", label: "Visão geral", icon: LayoutDashboard },
  { id: "camera", label: "Monitoramento", icon: Camera },
  { id: "people", label: "Pessoas e cadastros", icon: Users },
  { id: "alerts", label: "Alertas", icon: AlertTriangle },
  { id: "reports", label: "Relatórios", icon: BarChart3 },
  { id: "settings", label: "Configurações", icon: Settings },
];

const titles: Record<Tab, string> = {
  overview: "Visão geral",
  camera: "Monitoramento ao vivo",
  people: "Pessoas e cadastros",
  alerts: "Alertas",
  reports: "Relatórios",
  settings: "Configurações",
};

export default function Home() {
  const [tab, setTab] = useState<Tab>("overview");
  const [status, setStatus] = useState<Status>(initialStatus);
  const [apiOnline, setApiOnline] = useState(false);
  const [people, setPeople] = useState<string[]>([]);
  const [enrollmentName, setEnrollmentName] = useState("");
  const [deletingPerson, setDeletingPerson] = useState<string | null>(null);

  const [enrollment, setEnrollment] = useState<Enrollment>({
    active: false,
    name: null,
    samples: 0,
    required: 5,
    maximum: 12,
    message: "Nenhum cadastro em andamento.",
  });

  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);

  const [events, setEvents] = useState<EventRecord[]>([]);
  const [eventSummary, setEventSummary] = useState<EventSummary>({
    total: 0,
    active: 0,
    resolved: 0,
    today: 0,
  });

  const [eventFilter, setEventFilter] = useState<
    "all" | "active" | "resolved"
  >("all");

  const [eventSearch, setEventSearch] = useState("");
  const [eventsAvailable, setEventsAvailable] = useState(false);

  const refreshStatus = useCallback(async () => {
    try {
      const response = await fetch(`${API_URL}/api/status`, {
        cache: "no-store",
      });

      if (!response.ok) {
        throw new Error(`API respondeu com HTTP ${response.status}`);
      }

      const data = await response.json();

      setStatus({
        ...initialStatus,
        ...data,
        faces: Array.isArray(data.faces) ? data.faces : [],
      });

      setApiOnline(true);
      setLastUpdated(new Date());
    } catch {
      setApiOnline(false);

      setStatus((old) => ({
        ...old,
        environment: "API da AI BOX indisponível",
        camera_online: false,
        ai_online: false,
      }));
    }
  }, []);

  const refreshEvents = useCallback(async () => {
    try {
      const response = await fetch(
        `${API_URL}/api/events?limit=500&status=all`,
        { cache: "no-store" }
      );

      if (!response.ok) {
        throw new Error("Histórico indisponível");
      }

      const data = await response.json();

      setEvents(Array.isArray(data.events) ? data.events : []);

      setEventSummary({
        total: 0,
        active: 0,
        resolved: 0,
        today: 0,
        ...(data.summary || {}),
      });

      setEventsAvailable(true);
    } catch {
      setEventsAvailable(false);
    }
  }, []);

  const refreshPeopleAndEnrollment = useCallback(async () => {
    try {
      const [peopleResponse, enrollmentResponse] = await Promise.all([
        fetch(`${API_URL}/api/people`, { cache: "no-store" }),
        fetch(`${API_URL}/api/enrollment/status`, {
          cache: "no-store",
        }),
      ]);

      if (peopleResponse.ok) {
        const data = await peopleResponse.json();

        setPeople(
          Array.isArray(data.people) ? data.people : []
        );
      }

      if (enrollmentResponse.ok) {
        const data = await enrollmentResponse.json();

        setEnrollment((old) => ({
          ...old,
          ...data,
          required: data.required ?? old.required,
          maximum: data.maximum ?? old.maximum,
        }));
      }
    } catch {
      // A tela informa a situação da conexão pelo status principal.
    }
  }, []);

  useEffect(() => {
    void refreshStatus();
    void refreshPeopleAndEnrollment();
    void refreshEvents();

    const statusTimer = setInterval(
      () => void refreshStatus(),
      1500
    );

    const peopleTimer = setInterval(
      () => void refreshPeopleAndEnrollment(),
      3000
    );

    const eventsTimer = setInterval(
      () => void refreshEvents(),
      5000
    );

    return () => {
      clearInterval(statusTimer);
      clearInterval(peopleTimer);
      clearInterval(eventsTimer);
    };
  }, [
    refreshStatus,
    refreshPeopleAndEnrollment,
    refreshEvents,
  ]);

  const doEnrollment = async (
    action: "start" | "finish" | "cancel"
  ) => {
    setBusy(true);
    setError("");
    setNotice("");

    try {
      const response = await fetch(
        `${API_URL}/api/enrollment/${action}`,
        {
          method: "POST",
          headers:
            action === "start"
              ? { "Content-Type": "application/json" }
              : undefined,
          body:
            action === "start"
              ? JSON.stringify({ name: enrollmentName.trim() })
              : undefined,
        }
      );

      const data = await response.json().catch(() => ({}));

      if (!response.ok) {
        throw new Error(
          typeof data.detail === "string"
            ? data.detail
            : "Não foi possível concluir a operação."
        );
      }

      setNotice(
        data.message ||
          (action === "start"
            ? "Cadastro iniciado. Posicione uma pessoa diante da câmera."
            : action === "finish"
              ? "Cadastro finalizado."
              : "Cadastro cancelado.")
      );

      if (action === "finish") {
        setEnrollmentName("");
      }

      await refreshPeopleAndEnrollment();
    } catch (e) {
      setError(
        e instanceof Error
          ? e.message
          : "Falha de comunicação com a API."
      );
    } finally {
      setBusy(false);
    }
  };

  const deletePerson = async (person: string) => {
    const confirmed = window.confirm(
      `Tem certeza de que deseja excluir o cadastro de "${person}"?\n\nEssa ação removerá os embeddings faciais salvos e não poderá ser desfeita.`
    );
    if (!confirmed) return;

    setDeletingPerson(person);
    setError("");
    setNotice("");

    try {
      const response = await fetch(
        `${API_URL}/api/people/${encodeURIComponent(person)}`,
        { method: "DELETE", cache: "no-store" }
      );
      const data = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(
          typeof data.detail === "string"
            ? data.detail
            : "Não foi possível excluir este cadastro."
        );
      }

      setNotice(data.message || `Cadastro de ${person} excluído com sucesso.`);
      await refreshPeopleAndEnrollment();
      await refreshStatus();
    } catch (e) {
      setError(
        e instanceof Error
          ? e.message
          : "Falha de comunicação ao excluir o cadastro."
      );
    } finally {
      setDeletingPerson(null);
    }
  };

  const safe = status.unauthorized_count === 0;

  const connectionLabel = !apiOnline
    ? "API desconectada"
    : status.camera_online && status.ai_online
      ? "Sistema operacional"
      : "Sistema parcial";

  const visibleEvents = useMemo(
    () =>
      events.filter((event) => {
        const matchesStatus =
          eventFilter === "all" ||
          event.status === eventFilter;

        const q = eventSearch
          .trim()
          .toLocaleLowerCase("pt-BR");

        const matchesSearch =
          !q ||
          `${event.title} ${event.description} ${event.id}`
            .toLocaleLowerCase("pt-BR")
            .includes(q);

        return matchesStatus && matchesSearch;
      }),
    [events, eventFilter, eventSearch]
  );

  const exportEvents = () => {
    const columns = [
      "id",
      "title",
      "severity",
      "status",
      "first_seen_at",
      "confirmed_at",
      "resolved_at",
      "duration_seconds",
      "people_count",
      "description",
    ] as const;

    const escapeCsv = (value: unknown) =>
      `"${String(value ?? "").replaceAll('"', '""')}"`;

    const csv = [
      columns.join(";"),
      ...visibleEvents.map((event) =>
        columns
          .map((key) => escapeCsv(event[key]))
          .join(";")
      ),
    ].join("\r\n");

    const blob = new Blob(["\ufeff", csv], {
      type: "text/csv;charset=utf-8;",
    });

    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");

    link.href = url;
    link.download = `visionai-eventos-${new Date()
      .toISOString()
      .slice(0, 10)}.csv`;

    link.click();
    URL.revokeObjectURL(url);
  };

  return (
    <main className="shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">
            V<span>AI</span>
          </div>

          <div>
            <strong>VisionAI</strong>
            <small>INTELLIGENT MONITORING</small>
          </div>
        </div>

        <div className="workspace-label">PLATAFORMA</div>

        <nav
          className="navigation"
          aria-label="Navegação principal"
        >
          {nav.map(({ id, label, icon: Icon }) => (
            <button
              key={id}
              onClick={() => setTab(id)}
              className={`nav-item ${
                tab === id ? "active" : ""
              }`}
              aria-current={tab === id ? "page" : undefined}
            >
              <Icon size={18} />
              <span>{label}</span>

              {tab === id && (
                <ChevronRight
                  size={15}
                  className="nav-chevron"
                />
              )}
            </button>
          ))}
        </nav>

        <div className="sidebar-bottom">
          <div className="system-mini">
            <span
              className={`status-dot ${
                apiOnline &&
                status.camera_online &&
                status.ai_online
                  ? "online"
                  : "offline"
              }`}
            />

            <div>
              <strong>{connectionLabel}</strong>
              <small>VisionAI V2 · AI BOX</small>
            </div>
          </div>

          <div className="sidebar-foot">
            VISIONAI · GRUPO 13 <span>v2.0</span>
          </div>
        </div>
      </aside>

      <section className="main-area">
        <header className="topbar">
          <div>
            <div className="eyebrow">
              VISIONAI <span>/</span> PAINEL DE CONTROLE
            </div>

            <h1>{titles[tab]}</h1>

            <p className="subheading">
              Monitoramento inteligente e gestão de acessos
            </p>
          </div>

          <div className="top-actions">
            <div
              className={`connection-pill ${
                apiOnline ? "connected" : "disconnected"
              }`}
            >
              <span className="status-dot" />
              {connectionLabel}
            </div>

            <button
              className="icon-button"
              title="Atualizar dados"
              onClick={() => {
                void refreshStatus();
                void refreshPeopleAndEnrollment();
                void refreshEvents();
              }}
            >
              <RefreshCw size={17} />
            </button>
          </div>
        </header>

        <div className="content">
          {!apiOnline && (
            <div className="banner warning">
              <WifiOff size={18} />

              <div>
                <strong>
                  Não foi possível conectar à AI BOX
                </strong>

                <p>
                  Confira se a API está ativa em {API_URL}.
                  O painel tentará reconectar automaticamente.
                </p>
              </div>
            </div>
          )}

          {tab === "overview" && (
            <>
              <div
                className={`environment-banner ${
                  safe ? "is-safe" : "is-alert"
                }`}
              >
                <div className="environment-icon">
                  {safe ? (
                    <ShieldCheck size={25} />
                  ) : (
                    <AlertTriangle size={25} />
                  )}
                </div>

                <div className="environment-copy">
                  <span>SITUAÇÃO DO AMBIENTE</span>

                  <h2>
                    {apiOnline
                      ? status.environment
                      : "Aguardando conexão"}
                  </h2>

                  <p>
                    {!apiOnline
                      ? "Os dados serão atualizados quando a API estiver disponível."
                      : safe
                        ? "Nenhuma pessoa não autorizada detectada neste momento."
                        : "Atenção: há pessoa(s) não autorizada(s) detectada(s)."}
                  </p>
                </div>

                <div className="environment-tag">
                  {safe ? "NORMAL" : "ATENÇÃO"}
                </div>
              </div>

              <div className="metric-grid">
                <Metric
                  icon={Users}
                  label="Pessoas detectadas"
                  value={status.people_count}
                  detail="No campo de visão"
                />

                <Metric
                  icon={ShieldCheck}
                  label="Autorizadas"
                  value={status.authorized_count}
                  detail="Identificação confirmada"
                  tone="green"
                />

                <Metric
                  icon={AlertTriangle}
                  label="Não autorizadas"
                  value={status.unauthorized_count}
                  detail="Requer atenção"
                  tone={
                    status.unauthorized_count
                      ? "red"
                      : "default"
                  }
                />

                <Metric
                  icon={Camera}
                  label="Câmera principal"
                  value={
                    status.camera_online ? "ONLINE" : "OFFLINE"
                  }
                  detail="CAM-01"
                  tone={
                    status.camera_online ? "green" : "red"
                  }
                />
              </div>

              <div className="dashboard-grid">
                <Panel
                  title="Monitoramento ao vivo"
                  subtitle="Câmera principal · CAM-01"
                  action={
                    <button
                      className="text-button"
                      onClick={() => setTab("camera")}
                    >
                      Abrir monitoramento
                      <ChevronRight size={15} />
                    </button>
                  }
                >
                  <CameraPreview
                    apiUrl={API_URL}
                    online={status.camera_online}
                  />
                </Panel>

                <Panel
                  title="Desempenho do sistema"
                  subtitle="Métricas recebidas da API"
                >
                  <div className="performance-list">
                    <Performance
                      icon={Activity}
                      label="FPS da câmera"
                      value={`${status.fps ?? 0}`}
                    />

                    <Performance
                      icon={Cpu}
                      label="Tempo de inferência"
                      value={`${status.inference_ms ?? 0} ms`}
                    />

                    <Performance
                      icon={Camera}
                      label="Câmera"
                      value={
                        status.camera_online
                          ? "Conectada"
                          : "Desconectada"
                      }
                      good={status.camera_online}
                    />

                    <Performance
                      icon={Wifi}
                      label="Motor de IA"
                      value={
                        status.ai_online ? "Online" : "Offline"
                      }
                      good={status.ai_online}
                    />
                  </div>

                  <div className="updated-note">
                    {lastUpdated
                      ? `Última atualização às ${lastUpdated.toLocaleTimeString("pt-BR")}`
                      : "Aguardando primeira atualização"}
                  </div>
                </Panel>
              </div>

              <Panel
                title="Pessoas no campo de visão"
                subtitle="Identificações reportadas pela API"
                action={
                  <button
                    className="text-button"
                    onClick={() => setTab("people")}
                  >
                    Gerenciar pessoas
                    <ChevronRight size={15} />
                  </button>
                }
              >
                <FaceList faces={status.faces} />
              </Panel>
            </>
          )}

          {tab === "camera" && (
            <>
              <div className="section-intro">
                <div>
                  <h2>Câmera principal</h2>
                  <p>
                    Transmissão MJPEG da AI BOX com as detecções
                    reportadas pelo sistema.
                  </p>
                </div>

                <span
                  className={`state-label ${
                    status.camera_online ? "good" : "bad"
                  }`}
                >
                  {status.camera_online
                    ? "● ONLINE"
                    : "● OFFLINE"}
                </span>
              </div>

              <Panel
                title="Transmissão ao vivo"
                subtitle="CAM-01 · stream /video"
              >
                <CameraPreview
                  apiUrl={API_URL}
                  online={status.camera_online}
                  large
                />
              </Panel>

              <div className="metric-grid compact">
                <Metric
                  icon={Activity}
                  label="FPS"
                  value={status.fps}
                />

                <Metric
                  icon={Cpu}
                  label="Inferência"
                  value={`${status.inference_ms} ms`}
                />

                <Metric
                  icon={Users}
                  label="Pessoas"
                  value={status.people_count}
                />
              </div>
            </>
          )}

          {tab === "people" && (
            <>
              <div className="section-intro">
                <div>
                  <h2>Gerenciamento de pessoas</h2>
                  <p>
                    Cadastre pessoas pela câmera e acompanhe o
                    processo de captura facial.
                  </p>
                </div>

                <span className="count-pill">
                  {people.length} cadastrada(s)
                </span>
              </div>

              <div className="dashboard-grid people-grid">
                <Panel
                  title="Novo cadastro facial"
                  subtitle="Mantenha apenas uma pessoa diante da câmera durante a captura"
                >
                  <label
                    className="field-label"
                    htmlFor="person-name"
                  >
                    Nome completo
                  </label>

                  <input
                    id="person-name"
                    className="text-input"
                    value={enrollmentName}
                    onChange={(e) =>
                      setEnrollmentName(e.target.value)
                    }
                    disabled={enrollment.active || busy}
                    maxLength={80}
                    placeholder="Ex.: Gabriel Nagase"
                  />

                  <div className="capture-box">
                    <div className="capture-top">
                      <span>Amostras capturadas</span>
                      <strong>
                        {enrollment.samples}/{enrollment.maximum}
                      </strong>
                    </div>

                    <div className="progress-track">
                      <div
                        className="progress-value"
                        style={{
                          width: `${Math.min(
                            100,
                            (enrollment.samples /
                              Math.max(1, enrollment.maximum)) *
                              100
                          )}%`,
                        }}
                      />
                    </div>

                    <p>{enrollment.message}</p>

                    <small>
                      Mínimo para salvar: {enrollment.required}{" "}
                      amostras válidas.
                    </small>
                  </div>

                  {error && (
                    <div className="inline-message error">
                      <XCircle size={17} />
                      {error}
                    </div>
                  )}

                  {notice && (
                    <div className="inline-message success">
                      <CheckCircle2 size={17} />
                      {notice}
                    </div>
                  )}

                  <div className="button-row">
                    {!enrollment.active ? (
                      <button
                        className="primary-button"
                        disabled={
                          busy ||
                          !enrollmentName.trim() ||
                          !apiOnline
                        }
                        onClick={() =>
                          void doEnrollment("start")
                        }
                      >
                        {busy ? (
                          <LoaderCircle
                            className="spin"
                            size={17}
                          />
                        ) : (
                          <Plus size={17} />
                        )}
                        Iniciar cadastro
                      </button>
                    ) : (
                      <>
                        <button
                          className="primary-button"
                          disabled={
                            busy ||
                            enrollment.samples <
                              enrollment.required
                          }
                          onClick={() =>
                            void doEnrollment("finish")
                          }
                        >
                          {busy ? (
                            <LoaderCircle
                              className="spin"
                              size={17}
                            />
                          ) : (
                            <CheckCircle2 size={17} />
                          )}
                          Finalizar cadastro
                        </button>

                        <button
                          className="secondary-button danger-button"
                          disabled={busy}
                          onClick={() =>
                            void doEnrollment("cancel")
                          }
                        >
                          Cancelar
                        </button>
                      </>
                    )}
                  </div>
                </Panel>

                <Panel
                  title="Pessoas cadastradas"
                  subtitle="Lista retornada pelo banco facial"
                >
                  {error && (
                    <div className="inline-message error people-feedback">
                      <XCircle size={17} />
                      <span>{error}</span>
                    </div>
                  )}
                  {notice && (
                    <div className="inline-message success people-feedback">
                      <CheckCircle2 size={17} />
                      <span>{notice}</span>
                    </div>
                  )}
                  <div className="people-list">
                    {people.length ? (
                      people.map((person) => (
                        <div
                          className="person-row"
                          key={person}
                        >
                          <div className="person-avatar">
                            {person
                              .trim()
                              .charAt(0)
                              .toUpperCase()}
                          </div>

                          <div className="person-info">
                            <strong>{person}</strong>
                            <small>Cadastro facial</small>
                          </div>

                          <div className="person-actions">
                            <span className="state-label good">
                              CADASTRADA
                            </span>
                            <button
                              type="button"
                              className="delete-person-button"
                              onClick={() => void deletePerson(person)}
                              disabled={busy || deletingPerson !== null || enrollment.active}
                              title={`Excluir cadastro de ${person}`}
                              aria-label={`Excluir cadastro de ${person}`}
                            >
                              {deletingPerson === person ? (
                                <LoaderCircle className="spin" size={16} />
                              ) : (
                                <Trash2 size={16} />
                              )}
                              <span>{deletingPerson === person ? "Excluindo…" : "Excluir"}</span>
                            </button>
                          </div>
                        </div>
                      ))
                    ) : (
                      <Empty
                        icon={Users}
                        title="Nenhuma pessoa cadastrada"
                        text="Quando houver cadastros no banco facial, eles aparecerão aqui."
                      />
                    )}
                  </div>

                  <div className="people-footer">
                    <span>Total de pessoas</span>
                    <strong>{people.length}</strong>
                  </div>
                </Panel>
              </div>

              <Panel
                title="Prévia da câmera"
                subtitle="Confira o enquadramento antes de iniciar o cadastro"
              >
                <CameraPreview
                  apiUrl={API_URL}
                  online={status.camera_online}
                />
              </Panel>
            </>
          )}

          {tab === "alerts" && (
            <>
              <div className="metric-grid compact">
                <Metric
                  icon={AlertTriangle}
                  label="Incidentes registrados"
                  value={eventSummary.total}
                  detail="Histórico persistente"
                />

                <Metric
                  icon={Activity}
                  label="Ativos agora"
                  value={eventSummary.active}
                  detail="Ainda não resolvidos"
                  tone={
                    eventSummary.active ? "red" : "green"
                  }
                />

                <Metric
                  icon={CheckCircle2}
                  label="Resolvidos"
                  value={eventSummary.resolved}
                  detail="Ausência confirmada"
                  tone="green"
                />

                <Metric
                  icon={BarChart3}
                  label="Hoje"
                  value={eventSummary.today}
                  detail="Incidentes confirmados"
                />
              </div>

              <Panel
                title="Situação em tempo real"
                subtitle="Estado atual da câmera, separado do histórico persistente"
              >
                <div
                  className={`current-alert ${
                    status.unauthorized_count > 0
                      ? "alert-red"
                      : "alert-green"
                  }`}
                >
                  <div>
                    {status.unauthorized_count > 0 ? (
                      <AlertTriangle size={20} />
                    ) : (
                      <CheckCircle2 size={20} />
                    )}
                  </div>

                  <div>
                    <strong>
                      {status.unauthorized_count > 0
                        ? "Pessoa(s) não autorizada(s) no campo de visão"
                        : "Nenhuma pessoa não autorizada no último estado"}
                    </strong>

                    <p>
                      {status.unauthorized_count} pessoa(s) não
                      autorizada(s) neste momento. O registro
                      histórico só é confirmado após o tempo de
                      persistência configurado.
                    </p>
                  </div>
                </div>

                {!eventsAvailable && (
                  <div className="inline-message error">
                    <WifiOff size={16} />
                    O endpoint de histórico ainda não respondeu.
                    Integre `backend/event_store.py` à API e
                    reinicie o servidor; o restante do painel
                    continuará funcionando.
                  </div>
                )}
              </Panel>

              <Panel
                title="Histórico de incidentes"
                subtitle="Um registro por ocorrência contínua, sem gerar um alerta a cada frame"
                action={
                  <button
                    className="secondary-button"
                    onClick={() => void refreshEvents()}
                  >
                    <RefreshCw size={14} />
                    Atualizar
                  </button>
                }
              >
                <div className="event-toolbar">
                  <div className="event-search">
                    <Search size={16} />

                    <input
                      aria-label="Buscar incidentes"
                      placeholder="Buscar por título, descrição ou ID"
                      value={eventSearch}
                      onChange={(e) =>
                        setEventSearch(e.target.value)
                      }
                    />
                  </div>

                  <div className="event-actions">
                    <select
                      aria-label="Filtrar status"
                      value={eventFilter}
                      onChange={(e) =>
                        setEventFilter(
                          e.target.value as
                            | "all"
                            | "active"
                            | "resolved"
                        )
                      }
                    >
                      <option value="all">
                        Todos os status
                      </option>
                      <option value="active">Ativos</option>
                      <option value="resolved">Resolvidos</option>
                    </select>

                    <button
                      className="secondary-button"
                      onClick={exportEvents}
                      disabled={!visibleEvents.length}
                    >
                      <Download size={14} />
                      Exportar CSV
                    </button>
                  </div>
                </div>

                {!eventsAvailable ? (
                  <Empty
                    icon={AlertTriangle}
                    title="Histórico ainda não integrado"
                    text="A tela já está pronta para receber os eventos reais. Siga o guia backend/INTEGRAR_API.md para ativar o SQLite e os endpoints."
                  />
                ) : visibleEvents.length === 0 ? (
                  <Empty
                    icon={CheckCircle2}
                    title="Nenhum incidente encontrado"
                    text="Quando uma ocorrência não autorizada persistir por tempo suficiente, ela será registrada aqui."
                  />
                ) : (
                  <div className="event-table-wrap">
                    <table className="event-table">
                      <thead>
                        <tr>
                          <th>Incidente</th>
                          <th>Confirmado em</th>
                          <th>Duração</th>
                          <th>Pessoas</th>
                          <th>Status</th>
                        </tr>
                      </thead>

                      <tbody>
                        {visibleEvents.map((event) => (
                          <tr key={event.id}>
                            <td>
                              <div className="event-title">
                                <span
                                  className={`event-severity ${
                                    event.status === "active"
                                      ? "danger"
                                      : "resolved"
                                  }`}
                                />

                                <div>
                                  <strong>
                                    #{event.id} · {event.title}
                                  </strong>
                                  <small>{event.description}</small>
                                </div>
                              </div>
                            </td>

                            <td>
                              {formatDate(event.confirmed_at)}
                            </td>

                            <td>
                              {event.duration_seconds == null
                                ? event.status === "active"
                                  ? "Em andamento"
                                  : "—"
                                : `${Math.round(
                                    event.duration_seconds
                                  )} s`}
                            </td>

                            <td>{event.people_count}</td>

                            <td>
                              <span
                                className={`state-label ${
                                  event.status === "active"
                                    ? "bad"
                                    : "good"
                                }`}
                              >
                                {event.status === "active"
                                  ? "ATIVO"
                                  : "RESOLVIDO"}
                              </span>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}

                <p className="muted-note">
                  Banco local SQLite. O histórico permanece salvo
                  após reiniciar a API, desde que o arquivo
                  `data/visionai_events.sqlite3` seja mantido.
                </p>
              </Panel>
            </>
          )}

          {tab === "reports" && (
            <>
              <div className="section-intro">
                <div>
                  <h2>Relatório operacional</h2>
                  <p>
                    Indicadores atuais e histórico real de incidentes
                    persistidos.
                  </p>
                </div>

                <button
                  className="secondary-button"
                  onClick={exportEvents}
                  disabled={!events.length}
                >
                  <Download size={15} />
                  Exportar eventos CSV
                </button>
              </div>

              <div className="metric-grid">
                <Metric
                  icon={Users}
                  label="Pessoas no campo de visão"
                  value={status.people_count}
                  detail="Estado atual"
                />

                <Metric
                  icon={ShieldCheck}
                  label="Autorizadas"
                  value={status.authorized_count}
                  detail="Estado atual"
                  tone="green"
                />

                <Metric
                  icon={AlertTriangle}
                  label="Não autorizadas"
                  value={status.unauthorized_count}
                  detail="Estado atual"
                  tone={
                    status.unauthorized_count
                      ? "red"
                      : "default"
                  }
                />

                <Metric
                  icon={Activity}
                  label="Inferência"
                  value={`${status.inference_ms} ms`}
                  detail={`${status.fps} FPS reportados`}
                />
              </div>

              <div className="dashboard-grid">
                <Panel
                  title="Incidentes por situação"
                  subtitle="Contagem baseada no banco de eventos"
                >
                  <div className="report-bars">
                    {[
                      {
                        label: "Ativos",
                        value: eventSummary.active,
                        tone: "red",
                      },
                      {
                        label: "Resolvidos",
                        value: eventSummary.resolved,
                        tone: "green",
                      },
                      {
                        label: "Confirmados hoje",
                        value: eventSummary.today,
                        tone: "cyan",
                      },
                    ].map((item) => (
                      <div
                        className="report-bar-row"
                        key={item.label}
                      >
                        <div>
                          <span>{item.label}</span>
                          <strong>{item.value}</strong>
                        </div>

                        <div className="report-bar-track">
                          <span
                            className={`report-bar-fill ${item.tone}`}
                            style={{
                              width: `${
                                Math.min(
                                  100,
                                  eventSummary.total
                                    ? (item.value /
                                        eventSummary.total) *
                                        100
                                    : 0
                                )
                              }%`,
                            }}
                          />
                        </div>
                      </div>
                    ))}
                  </div>

                  {!eventsAvailable && (
                    <p className="muted-note">
                      Ative o endpoint `/api/events` para preencher
                      os indicadores históricos. Nenhum dado
                      fictício é exibido.
                    </p>
                  )}
                </Panel>

                <Panel
                  title="Resumo do período"
                  subtitle="Eventos registrados no banco"
                >
                  <div className="report-summary">
                    <div>
                      <span>Total histórico</span>
                      <strong>{eventSummary.total}</strong>
                    </div>

                    <div>
                      <span>Confirmados hoje</span>
                      <strong>{eventSummary.today}</strong>
                    </div>

                    <div>
                      <span>Em aberto</span>
                      <strong>{eventSummary.active}</strong>
                    </div>

                    <div>
                      <span>Resolvidos</span>
                      <strong>{eventSummary.resolved}</strong>
                    </div>
                  </div>

                  <p className="muted-note">
                    Última leitura do status:{" "}
                    {lastUpdated
                      ? lastUpdated.toLocaleString("pt-BR")
                      : "aguardando conexão"}
                    .
                  </p>
                </Panel>
              </div>

              <Panel
                title="Eventos recentes"
                subtitle="Últimos registros do histórico"
              >
                <div className="face-list">
                  {events.slice(0, 5).map((event) => (
                    <div
                      className="face-row"
                      key={event.id}
                    >
                      <div
                        className={`face-status-icon ${
                          event.status === "active"
                            ? "bad"
                            : "good"
                        }`}
                      >
                        {event.status === "active" ? (
                          <AlertTriangle size={18} />
                        ) : (
                          <CheckCircle2 size={18} />
                        )}
                      </div>

                      <div className="face-details">
                        <strong>
                          #{event.id} · {event.title}
                        </strong>
                        <small>
                          {formatDate(event.confirmed_at)} ·{" "}
                          {event.status === "active"
                            ? "Em andamento"
                            : "Resolvido"}
                        </small>
                      </div>

                      <span
                        className={`state-label ${
                          event.status === "active"
                            ? "bad"
                            : "good"
                        }`}
                      >
                        {event.status === "active"
                          ? "ATIVO"
                          : "RESOLVIDO"}
                      </span>
                    </div>
                  ))}

                  {eventsAvailable && !events.length && (
                    <Empty
                      icon={BarChart3}
                      title="Sem dados históricos"
                      text="Os incidentes aparecerão após a primeira ocorrência confirmada."
                    />
                  )}

                  {!eventsAvailable && (
                    <Empty
                      icon={WifiOff}
                      title="API de eventos não conectada"
                      text="Os dados em tempo real continuam disponíveis; integre o módulo de histórico para relatórios persistentes."
                    />
                  )}
                </div>
              </Panel>
            </>
          )}

          {tab === "settings" && (
            <Placeholder
              icon={Settings}
              title="Configurações do painel"
              text="Configurações da interface e conexão. O modelo, os limiares de reconhecimento e a câmera são configurados no backend e não serão alterados por esta tela."
            >
              <div className="settings-row">
                <div>
                  <strong>Endereço da API</strong>
                  <p>
                    Origem usada pelo dashboard para consultar o
                    VisionAI.
                  </p>
                </div>
                <code>{API_URL}</code>
              </div>

              <div className="settings-row">
                <div>
                  <strong>Atualização do status</strong>
                  <p>Frequência de consulta do painel.</p>
                </div>
                <span className="state-label">1,5 segundo</span>
              </div>

              <div className="settings-row">
                <div>
                  <strong>Conexão com a API</strong>
                  <p>Estado da última consulta.</p>
                </div>
                <span
                  className={`state-label ${
                    apiOnline ? "good" : "bad"
                  }`}
                >
                  {apiOnline ? "CONECTADA" : "DESCONECTADA"}
                </span>
              </div>

              <div className="settings-help">
                <CircleHelp size={18} />

                <span>
                  Para mudar o endereço, configure{" "}
                  <code>NEXT_PUBLIC_VISIONAI_API_URL</code> no
                  arquivo <code>.env.local</code> do dashboard.
                </span>
              </div>
            </Placeholder>
          )}

          <footer className="page-footer">
            <span>
              VisionAI V2 · Sistema de monitoramento inteligente
            </span>
            <span>
              Dados em tempo real quando a API está conectada
            </span>
          </footer>
        </div>
      </section>
    </main>
  );
}

function formatDate(value: string | null | undefined) {
  if (!value) return "—";

  const date = new Date(value);

  return Number.isNaN(date.getTime())
    ? value
    : date.toLocaleString("pt-BR", {
        dateStyle: "short",
        timeStyle: "medium",
      });
}

function Metric({
  icon: Icon,
  label,
  value,
  detail,
  tone = "default",
}: {
  icon: typeof Users;
  label: string;
  value: string | number;
  detail?: string;
  tone?: "default" | "green" | "red";
}) {
  return (
    <div className="metric-card">
      <div className={`metric-icon ${tone}`}>
        <Icon size={19} />
      </div>

      <span className="metric-label">{label}</span>
      <strong className={`metric-value ${tone}`}>{value}</strong>

      {detail && <small>{detail}</small>}
    </div>
  );
}

function Panel({
  title,
  subtitle,
  action,
  children,
}: {
  title: string;
  subtitle?: string;
  action?: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <section className="panel">
      <div className="panel-heading">
        <div>
          <h3>{title}</h3>
          {subtitle && <p>{subtitle}</p>}
        </div>

        {action}
      </div>

      {children}
    </section>
  );
}

function CameraPreview({
  apiUrl,
  online,
  large = false,
}: {
  apiUrl: string;
  online: boolean;
  large?: boolean;
}) {
  const [failed, setFailed] = useState(false);
  const [streamAttempt, setStreamAttempt] = useState(0);

  // Quando a câmera volta a ficar online, tenta reconectar o stream
  useEffect(() => {
    if (online && failed) {
      setFailed(false);
      setStreamAttempt((a) => a + 1);
    }
  }, [online, failed]);

  // Reconexão automática a cada 5s enquanto estiver com falha
  useEffect(() => {
    if (!failed) return;
    const timer = setTimeout(() => {
      setFailed(false);
      setStreamAttempt((a) => a + 1);
    }, 5000);
    return () => clearTimeout(timer);
  }, [failed]);

  const src = `${apiUrl}/video?t=${streamAttempt}`;

  return (
    <div className={`camera-frame ${large ? "camera-large" : ""}`}>
      {!failed ? (
        <img
          key={src}
          src={src}
          alt="Transmissão ao vivo da câmera VisionAI"
          onError={(event) => {
            console.error("Erro ao carregar o vídeo da VisionAI:", {
              url: event.currentTarget.src,
              timestamp: new Date().toISOString(),
            });
            setFailed(true);
          }}
        />
      ) : (
        <div className="camera-placeholder">
          <Camera size={32} />
          <strong>Transmissão indisponível</strong>
          <span>Tentando reconectar automaticamente...</span>
          <button
            className="secondary-button"
            onClick={() => {
              setFailed(false);
              setStreamAttempt((a) => a + 1);
            }}
          >
            Tentar novamente
          </button>
        </div>
      )}

      <div className="camera-overlay">
        <span className="live-dot" />
        LIVE · CAM-01
      </div>

      {!online && (
        <div className="camera-offline">
          Câmera offline ou aguardando conexão
        </div>
      )}
    </div>
  );
}

function FaceList({ faces }: { faces: Face[] }) {
  if (!faces.length) {
    return (
      <Empty
        icon={Users}
        title="Nenhum rosto detectado"
        text="As detecções aparecerão aqui quando a câmera identificar pessoas."
      />
    );
  }

  return (
    <div className="face-list">
      {faces.map((face, i) => (
        <div
          className="face-row"
          key={`${face.name}-${i}`}
        >
          <div
            className={`face-status-icon ${
              face.authorized ? "good" : "bad"
            }`}
          >
            {face.authorized ? (
              <CheckCircle2 size={18} />
            ) : (
              <AlertTriangle size={18} />
            )}
          </div>

          <div className="face-details">
            <strong>
              {face.authorized
                ? face.name
                : "Pessoa não autorizada"}
            </strong>

            <small>
              Confiança:{" "}
              {Number(face.score || 0).toLocaleString("pt-BR", {
                style: "percent",
                maximumFractionDigits: 1,
              })}
            </small>
          </div>

          <span
            className={`state-label ${
              face.authorized ? "good" : "bad"
            }`}
          >
            {face.authorized
              ? "AUTORIZADA"
              : "NÃO AUTORIZADA"}
          </span>
        </div>
      ))}
    </div>
  );
}

function Performance({
  icon: Icon,
  label,
  value,
  good,
}: {
  icon: typeof Activity;
  label: string;
  value: string;
  good?: boolean;
}) {
  return (
    <div className="performance-row">
      <span className="performance-icon">
        <Icon size={17} />
      </span>

      <span>{label}</span>

      <strong
        className={
          good === undefined
            ? ""
            : good
              ? "green-text"
              : "red-text"
        }
      >
        {value}
      </strong>
    </div>
  );
}

function Empty({
  icon: Icon,
  title,
  text,
}: {
  icon: typeof Users;
  title: string;
  text: string;
}) {
  return (
    <div className="empty-state">
      <Icon size={26} />
      <strong>{title}</strong>
      <p>{text}</p>
    </div>
  );
}

function Placeholder({
  icon: Icon,
  title,
  text,
  children,
}: {
  icon: typeof Settings;
  title: string;
  text: string;
  children: React.ReactNode;
}) {
  return (
    <Panel title={title} subtitle={text}>
      <div className="placeholder-content">
        <div className="placeholder-icon">
          <Icon size={25} />
        </div>
        {children}
      </div>
    </Panel>
  );
}
