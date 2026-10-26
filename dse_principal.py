import concurrent.futures
import os
import random
import re
import subprocess
import time
import xml.etree.ElementTree as ET

BASE_DIR = os.path.abspath(".")
BINARIO_GEM5 = os.path.join(BASE_DIR, "build/ARM/gem5.fast")
SCRIPT_CORTEX = os.path.join(BASE_DIR, "scripts/CortexA76_scripts_gem5/CortexA76.py")
WORKLOAD_PATH = os.path.join(BASE_DIR, "workloads/jpeg2k_dec/jpg2k_dec")
WORKLOAD_INPUT = os.path.join(BASE_DIR, "workloads/jpeg2k_dec/jpg2kdec_testfile.j2k")

PREDICTOR_TIPO = "8"
SCRIPT_CONVERSOR_MCPAT = os.path.join(BASE_DIR, "scripts/McPAT/gem5toMcPAT_cortexA76.py")
PLANTILLA_XML = os.path.join(BASE_DIR, "scripts/McPAT/ARM_A76_2.1GHz.xml")
BINARIO_MCPAT = "/home/administrador/mcpat/mcpat"

CSV_SALIDA = os.path.join(BASE_DIR, "dse_1000_completo.csv")
PROFILING_SALIDA = os.path.join(BASE_DIR, "profiling_inicial.txt")

MAX_WORKERS = 10
TOTAL_EVALUACIONES = 1020
POBLACION_INICIAL = 30
MUTATION_RATE = 0.30
RANDOM_SEED = 42

SPACE = {
    "rob": [32, 64, 96, 128, 160, 192, 224, 256],
    "iq": [16, 24, 32, 48, 64, 80, 96, 128],
    "l2": ["256kB", "512kB", "1MB", "2MB"],
    "l1i": ["16kB", "32kB", "64kB", "128kB"],
    "l1d": ["16kB", "32kB", "64kB", "128kB"],
    "width": [2, 4, 6, 8],
}
GENES = list(SPACE.keys())


def clave_config(cfg):
    return (
        int(cfg["rob"]),
        int(cfg["iq"]),
        str(cfg["l2"]),
        str(cfg["l1i"]),
        str(cfg["l1d"]),
        int(cfg["width"]),
    )


def individuo_aleatorio():
    return {k: random.choice(SPACE[k]) for k in GENES}


def cruzar(padre1, padre2):
    return {k: (padre1[k] if random.random() < 0.5 else padre2[k]) for k in GENES}


def mutar(cfg, tasa=MUTATION_RATE):
    nuevo = {k: cfg[k] for k in GENES}
    for k in GENES:
        if random.random() < tasa:
            opciones = SPACE[k]
            idx = opciones.index(nuevo[k])
            if random.random() < 0.7:
                vecinos = [i for i in (idx - 1, idx + 1) if 0 <= i < len(opciones)]
                if vecinos:
                    nuevo[k] = opciones[random.choice(vecinos)]
            else:
                nuevo[k] = random.choice(opciones)
    return nuevo


def generar_hijo_evolutivo(historial_valido, visitados):
    if len(historial_valido) < 6:
        while True:
            cand = individuo_aleatorio()
            k = clave_config(cand)
            if k not in visitados:
                visitados.add(k)
                return cand

    mejor_edp = min(historial_valido, key=lambda x: x["edp_cpi"])
    mejor_cpi = min(historial_valido, key=lambda x: x["cpi"])
    mejor_ene = min(historial_valido, key=lambda x: x["energia_cpi"])
    top_edp = sorted(historial_valido, key=lambda x: x["edp_cpi"])[:40]
    pool_padres = top_edp + [mejor_edp, mejor_cpi, mejor_ene]

    intentos = 0
    while True:
        intentos += 1
        if intentos > 500:
            hijo = individuo_aleatorio()
        else:
            p1 = min(
                random.sample(pool_padres, min(4, len(pool_padres))),
                key=lambda x: x["edp_cpi"],
            )
            p2 = min(
                random.sample(pool_padres, min(4, len(pool_padres))),
                key=lambda x: x["edp_cpi"],
            )
            hijo = cruzar(p1, p2)
            hijo = mutar(hijo, tasa=MUTATION_RATE)
            if clave_config(hijo) in visitados:
                g_azar = random.choice(GENES)
                hijo[g_azar] = random.choice(SPACE[g_azar])

        k = clave_config(hijo)
        if k not in visitados:
            visitados.add(k)
            return hijo


def hacer_profiling_inicial():
    stats_base = os.path.join(BASE_DIR, "m5out_dse_exp1/stats.txt")
    if not os.path.exists(stats_base):
        for root, dirs, files in os.walk(BASE_DIR):
            if "stats.txt" in files:
                stats_base = os.path.join(root, "stats.txt")
                break

    if not os.path.exists(stats_base):
        return

    with open(stats_base, "r") as f, open(PROFILING_SALIDA, "w") as out:
        out.write("====================================================\n")
        out.write("      PROFILING INICIAL DE LA CARGA JPEG2K_DEC\n")
        out.write("====================================================\n\n")

        inst_type_lines, fu_lines, cache_lines, branch_lines = [], [], [], []
        for line in f:
            if "op_class" in line or "statExecutedInstType" in line:
                inst_type_lines.append(line.strip())
            elif "fuBusy" in line or "fu_busy" in line:
                fu_lines.append(line.strip())
            elif "overallMissRate" in line or "overallHits" in line or "overallMisses" in line:
                if "l1" in line or "l2" in line or "icache" in line or "dcache" in line:
                    cache_lines.append(line.strip())
            elif "branchPred" in line or "condPredicted" in line or "condIncorrect" in line:
                branch_lines.append(line.strip())

        out.write("1. MEZCLA DE INSTRUCCIONES (OP CLASSES):\n")
        for l in inst_type_lines[:15]:
            out.write(f"  {l}\n")
        out.write("\n2. TASA DE FALLOS EN CACHÉ:\n")
        for l in cache_lines[:15]:
            out.write(f"  {l}\n")
        out.write("\n3. RENDIMIENTO PREDICCIÓN DE SALTOS:\n")
        for l in branch_lines[:15]:
            out.write(f"  {l}\n")
        out.write("\n4. UNIDADES FUNCIONALES Y OCUPACIÓN (FUs):\n")
        for l in fu_lines[:15]:
            out.write(f"  {l}\n")


def ejecutar_experimento(exp):
    i = exp["id"]
    gen = exp["gen"]
    rob = exp["rob"]
    iq = exp["iq"]
    l2 = exp["l2"]
    l1i = exp["l1i"]
    l1d = exp["l1d"]
    w = exp["width"]

    out_dir = os.path.join(BASE_DIR, f"m5out_1000_exp{i}")
    os.makedirs(out_dir, exist_ok=True)

    stats = os.path.join(out_dir, "stats.txt")
    cfg = os.path.join(out_dir, "config.json")
    xml_path = os.path.join(out_dir, "mcpat_input.xml")
    mout_path = os.path.join(out_dir, "mcpat_out.txt")
    img_out = os.path.join(out_dir, "image.pgm")

    t_inicio_total = time.time()

    t0_gem5 = time.time()
    if not os.path.exists(stats) or os.path.getsize(stats) == 0:
        workload_opts = f"-i {WORKLOAD_INPUT} -o {img_out}"
        cmd_gem5 = [
            BINARIO_GEM5, "-d", out_dir, SCRIPT_CORTEX,
            "-c", WORKLOAD_PATH, "-o", workload_opts,
            f"--branch_predictor_type={PREDICTOR_TIPO}",
            f"--rob_entries={rob}", f"--iq_entries={iq}",
            f"--l1i_size={l1i}", f"--l1d_size={l1d}", f"--l2_size={l2}",
            f"--fetch_width={w}", f"--decode_width={w}", f"--commit_width={w}"
        ]
        try:
            subprocess.run(cmd_gem5, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            return None
    t_gem5 = time.time() - t0_gem5

    if os.path.exists(img_out):
        try:
            os.remove(img_out)
        except OSError:
            pass

    ipc, cpi, sim_sec = 0.0, 0.0, 0.0
    if os.path.exists(stats):
        with open(stats, "r") as st:
            for line in st:
                if "system.cpu.ipc" in line and ipc == 0.0:
                    p = line.split()
                    if len(p) >= 2:
                        ipc = float(p[1])
                elif "system.cpu.cpi" in line and cpi == 0.0:
                    p = line.split()
                    if len(p) >= 2:
                        cpi = float(p[1])
                elif "simSeconds" in line and sim_sec == 0.0:
                    p = line.split()
                    if len(p) >= 2:
                        sim_sec = float(p[1])

    if cpi == 0.0 and ipc > 0.0:
        cpi = 1.0 / ipc

    t0_mcpat = time.time()
    leak, dyn = 0.0, 0.0
    if os.path.exists(stats) and os.path.exists(cfg):
        try:
            if not os.path.exists(mout_path) or os.path.getsize(mout_path) == 0:
                subprocess.run(
                    ["python3", SCRIPT_CONVERSOR_MCPAT, stats, cfg, PLANTILLA_XML],
                    cwd=out_dir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
                )
                archivo_gen = os.path.join(out_dir, "config.xml")
                if os.path.exists(archivo_gen):
                    os.replace(archivo_gen, xml_path)

                if os.path.exists(xml_path):
                    ET.parse(xml_path)
                    with open(mout_path, "w") as f_m:
                        subprocess.run(
                            [BINARIO_MCPAT, "-infile", xml_path, "-print_level", "1"],
                            stdout=f_m, stderr=subprocess.PIPE, text=True, check=True
                        )

            if os.path.exists(mout_path):
                with open(mout_path, "r") as f_m:
                    txt = f_m.read()
                    m_sub = re.search(r"Subthreshold Leakage\s*=\s*([\d\.]+)\s*W", txt)
                    m_gate = re.search(r"Gate Leakage\s*=\s*([\d\.]+)\s*W", txt)
                    m_tot = re.search(r"Total Leakage\s*=\s*([\d\.]+)\s*W", txt)
                    m_d = re.search(r"Runtime Dynamic\s*=\s*([\d\.]+)\s*W", txt)

                    if m_tot:
                        leak = float(m_tot.group(1))
                    else:
                        leak = (float(m_sub.group(1)) if m_sub else 0.0) + (float(m_gate.group(1)) if m_gate else 0.0)
                    if m_d:
                        dyn = float(m_d.group(1))
        except Exception:
            pass
    t_mcpat = time.time() - t0_mcpat

    pot_total = leak + dyn
    energia_j = pot_total * sim_sec
    edp_js = energia_j * sim_sec
    energia_cpi = pot_total * cpi
    edp_cpi = energia_cpi * cpi
    t_sim_real = time.time() - t_inicio_total

    return {
        "id": i, "gen": gen, "rob": rob, "iq": iq, "l2": l2, "l1i": l1i, "l1d": l1d, "width": w,
        "ipc": ipc, "cpi": cpi, "tiempo": sim_sec, "leak": leak, "dyn": dyn,
        "energia_j": energia_j, "edp_js": edp_js, "energia_cpi": energia_cpi, "edp_cpi": edp_cpi,
        "t_gem5": t_gem5, "t_mcpat": t_mcpat, "t_real": t_sim_real
    }


def main():
    random.seed(RANDOM_SEED)
    total_espacio = 1
    for v in SPACE.values():
        total_espacio *= len(v)

    print("======================================================================")
    print(" DSE EVOLUTIVO ASÍNCRONO (STEADY-STATE GA - 10 NÚCLEOS)")
    print(f" Espacio total de diseño: {total_espacio} combinaciones")
    print(f" Meta de simulaciones: {TOTAL_EVALUACIONES} en {MAX_WORKERS} núcleos")
    print("======================================================================\n")

    hacer_profiling_inicial()

    visitados = set()
    historial_valido = []
    completados = 0
    siguiente_id = 1

    if os.path.exists(CSV_SALIDA) and os.path.getsize(CSV_SALIDA) > 0:
        with open(CSV_SALIDA, "r") as f:
            lineas = f.readlines()
            for linea in lineas[1:]:
                partes = linea.strip().split(",")
                if len(partes) >= 20:
                    try:
                        rec = {
                            "id": int(partes[0]), "gen": int(partes[1]), "rob": int(partes[2]),
                            "iq": int(partes[3]), "l2": partes[4], "l1i": partes[5], "l1d": partes[6],
                            "width": int(partes[7]), "ipc": float(partes[8]), "cpi": float(partes[9]),
                            "tiempo": float(partes[10]), "leak": float(partes[11]), "dyn": float(partes[12]),
                            "energia_j": float(partes[13]), "edp_js": float(partes[14]),
                            "energia_cpi": float(partes[15]), "edp_cpi": float(partes[16])
                        }
                        visitados.add(clave_config(rec))
                        if rec["edp_cpi"] > 0 and rec["cpi"] > 0:
                            historial_valido.append(rec)
                        completados += 1
                        siguiente_id = max(siguiente_id, rec["id"] + 1)
                    except ValueError:
                        pass
        print(f" [AUTO-RESUME] Se recuperaron {completados} simulaciones previas desde `{CSV_SALIDA}`.")
    else:
        with open(CSV_SALIDA, "w") as f:
            f.write("Exp,Generacion,ROB,IQ,L2,L1i,L1d,Width,IPC,CPI,Tiempo_s,Pot_Est_W,Pot_Dyn_W,"
                    "Energia_J,EDP_Js,Energia_CPI,EDP_CPI,Tiempo_Gem5_s,Tiempo_McPAT_s,Tiempo_Sim_Real_s\n")

    if completados >= TOTAL_EVALUACIONES:
        print(" Ya se alcanzaron las simulaciones objetivo.")
        return

    cola_semillas = []
    extremo_min = {k: SPACE[k][0] for k in GENES}
    extremo_max = {k: SPACE[k][-1] for k in GENES}
    for ext in (extremo_min, extremo_max):
        k = clave_config(ext)
        if k not in visitados:
            visitados.add(k)
            cola_semillas.append(ext)

    while len(cola_semillas) + completados < POBLACION_INICIAL:
        cand = individuo_aleatorio()
        k = clave_config(cand)
        if k not in visitados:
            visitados.add(k)
            cola_semillas.append(cand)

    def despachar_siguiente():
        nonlocal siguiente_id
        if cola_semillas:
            ind = cola_semillas.pop(0)
        else:
            ind = generar_hijo_evolutivo(historial_valido, visitados)
        ind["id"] = siguiente_id
        ind["gen"] = ((siguiente_id - 1) // 30) + 1
        siguiente_id += 1
        return ind

    t_inicio_global = time.time()
    lanzados = completados

    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futuros_activos = set()

        while len(futuros_activos) < MAX_WORKERS and lanzados < TOTAL_EVALUACIONES:
            exp = despachar_siguiente()
            futuros_activos.add(executor.submit(ejecutar_experimento, exp))
            lanzados += 1

        while futuros_activos:
            hechos, futuros_activos = concurrent.futures.wait(
                futuros_activos, return_when=concurrent.futures.FIRST_COMPLETED
            )

            for fut in hechos:
                res = fut.result()
                completados += 1

                if res:
                    if res["edp_cpi"] > 0 and res["cpi"] > 0:
                        historial_valido.append(res)

                    with open(CSV_SALIDA, "a") as f:
                        f.write(f"{res['id']},{res['gen']},{res['rob']},{res['iq']},{res['l2']},{res['l1i']},{res['l1d']},{res['width']},"
                                f"{res['ipc']:.6f},{res['cpi']:.6f},{res['tiempo']:.6f},{res['leak']:.4f},{res['dyn']:.4f},"
                                f"{res['energia_j']:.6f},{res['edp_js']:.8f},{res['energia_cpi']:.6f},{res['edp_cpi']:.6f},"
                                f"{res['t_gem5']:.2f},{res['t_mcpat']:.2f},{res['t_real']:.2f}\n")

                    restantes = TOTAL_EVALUACIONES - completados
                    t_prom_s = res["t_real"]
                    eta_h = (restantes / MAX_WORKERS) * (t_prom_s / 3600)

                    mejor_edp_str = f"{min(x['edp_cpi'] for x in historial_valido):.4f}" if historial_valido else "N/A"
                    print(f"[{completados:04d}/{TOTAL_EVALUACIONES} | Gen {res['gen']:02d}] "
                          f"✔ Exp {res['id']} ({res['t_real']/60:.1f}m) | ROB={res['rob']} IQ={res['iq']} L2={res['l2']} W={res['width']} | "
                          f"CPI={res['cpi']:.4f} | EDP_CPI={res['edp_cpi']:.4f} (Récord: {mejor_edp_str}) | ETA: ~{eta_h:.1f}h", flush=True)

                if lanzados < TOTAL_EVALUACIONES:
                    nuevo_exp = despachar_siguiente()
                    futuros_activos.add(executor.submit(ejecutar_experimento, nuevo_exp))
                    lanzados += 1

    t_total = time.time() - t_inicio_global
    print("\n======================================================================")
    print("¡EXPLORACIÓN EVOLUTIVA DSE COMPLETADA CON ÉXITO!")
    print(f"Tiempo total acumulado: {t_total/3600:.2f} horas")
    print(f"Resultados guardados en `{CSV_SALIDA}`")
    print("======================================================================")


if __name__ == "__main__":
    main()