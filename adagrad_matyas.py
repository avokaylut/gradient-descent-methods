import numpy as np

from matplotlib import pyplot as plt, animation

def compute_gradient(f, x, h=1e-7):
    """Вычисление градиента функции F в точке x"""
    n = len(x)
    grad = np.zeros(n)
    for i in range(n):
        e_i = np.zeros(n)
        e_i[i] = 1.0
        x_plus = x + h * e_i
        x_minus = x - h * e_i
        grad[i] = (f(x_plus) - f(x_minus)) / (2 * h)
    return grad

def adagrad_detailed(f, x0, epsilon=1e-4, eta=0.1):
    """
    Adagrad с выводом подробной информации на каждой итерации
    и двумя критериями остановки:
    1. δ₁ = max_i(|x_i^{k+1} - x_i^k| / |x_i^k|)
    2. δ₂ = |f(X^{k+1}) - f(X^k)| / |f(X^k)|
    """
    x_j = np.asarray(x0, dtype=float).copy()
    j = 0

    G = np.zeros_like(x_j)
    epsilon_ada = 1e-8

    # Сохраняем историю для визуализации
    history_x = [x0.copy()]

    print("=" * 90)
    print(f"{'Итерация':^8} | {'x':^20} | {'F(x)':^10} | {'η адапт.':^10} | {'δ₁':^10} | {'δ₂':^10}")
    print("=" * 90)

    while True:
        if j >= 100000:
            print("Достигнут предел 100000 итераций")
            break
        # Вычисление градиента
        grad_F = compute_gradient(f, x_j)

        # Обновление Adagrad
        G += grad_F ** 2
        adaptive_lr = eta / (np.sqrt(G) + epsilon_ada)
        x_j_plus_1 = x_j - adaptive_lr * grad_F

        if not np.isfinite(x_j_plus_1).all():
            raise ValueError("Расходящиеся итерации: уменьшите eta")
        history_x.append(x_j_plus_1.copy())

        # Вычисление новых критериев остановки
        # 1. δ₁ = max_i(|Δx_i|/|x_i|)
        delta1_values = []
        for i in range(len(x_j)):
            if abs(x_j[i]) > 1e-10:
                delta1 = abs(x_j_plus_1[i] - x_j[i]) / abs(x_j[i])
            else:
                delta1 = abs(x_j_plus_1[i] - x_j[i])
            delta1_values.append(delta1)
        delta1 = max(delta1_values)

        # 2. δ₂ = |Δf|/|f|
        f_current = f(x_j)
        f_next = f(x_j_plus_1)
        if abs(f_current) > 1e-10:
            delta2 = abs(f_next - f_current) / abs(f_current)
        else:
            delta2 = abs(f_next - f_current)


        # Вывод информации на текущей итерации
        if j % 50 == 0 or delta1 < epsilon or delta2 < epsilon or j<10:
            # Форматируем вывод с 4 знаками после запятой
            x_formatted = f"[{x_j_plus_1[0]:.4f}, {x_j_plus_1[1]:.4f}]" if len(x_j_plus_1) == 2 else str(
                x_j_plus_1.round(4))
            lr_formatted = f"[{adaptive_lr[0]:.4f}, {adaptive_lr[1]:.4f}]" if len(adaptive_lr) == 2 else str(
                adaptive_lr.round(4))

            print(f"{j:^8} | {x_formatted:^20} | {f_next:^10.4f} | "
                  f"{lr_formatted:^10} | {delta1:^10.4f} | "
                  f"{delta2:^10.4f} ")

        # Проверка критериев остановки
        if delta1 < epsilon:
            print(f"\n✓ ОСТАНОВКА по δ₁: {delta1:.4f} < ε = {epsilon:.0e}")
            break
        elif delta2 < epsilon:
            print(f"\n✓ ОСТАНОВКА по δ₂: {delta2:.4f} < ε = {epsilon:.0e}")
            break


        # Переход к следующей итерации
        x_j = x_j_plus_1
        j += 1

    print(f"\nРЕЗУЛЬТАТ:")
    print(f"Начальная точка: [{x0[0]:.4f}, {x0[1]:.4f}]")
    print(f"Найденный минимум: x* = [{x_j_plus_1[0]:.4f}, {x_j_plus_1[1]:.4f}]")
    print(f"Значение функции: F(x*) = {f(x_j_plus_1):.4f}")
    print(f"Количество итераций: {len(history_x) - 1}")
    print("=" * 90)

    return x_j_plus_1, len(history_x) - 1, adaptive_lr, G, np.array(history_x)

def matyas_function(x):
    return 0.26 * (x[0] ** 2 + x[1] ** 2) - 0.48 * x[0] * x[1]

def plot_trajectory(f, history_x, x0, x_opt, eta):
    """График траектории приближения к минимуму"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Подготовка данных для контурного графика
    # Определяем границы по истории
    x_min = history_x[:, 0].min() - 1
    x_max = history_x[:, 0].max() + 1
    y_min = history_x[:, 1].min() - 1
    y_max = history_x[:, 1].max() + 1

    # Делаем квадратные границы для равного масштаба
    x_range = max(abs(x_min), abs(x_max))
    y_range = max(abs(y_min), abs(y_max))
    x_min, x_max = -x_range - 1, x_range + 1
    y_min, y_max = -y_range - 1, y_range + 1

    x_range_vals = np.linspace(x_min, x_max, 200)
    y_range_vals = np.linspace(y_min, y_max, 200)
    X, Y = np.meshgrid(x_range_vals, y_range_vals)
    Z = np.array([[f(np.array([x, y])) for x in x_range_vals] for y in y_range_vals])

    # 1. Контурный график с траекторией
    levels = np.linspace(Z.min(), Z.max(), 30)
    contour = ax1.contourf(X, Y, Z, levels=levels, cmap='viridis', alpha=0.7)
    ax1.contour(X, Y, Z, levels=levels, colors='black', alpha=0.3, linewidths=0.5)

    # Траектория
    ax1.plot(history_x[:, 0], history_x[:, 1], 'ro-', markersize=4,
             linewidth=1.5, alpha=0.7)
    ax1.plot(x0[0], x0[1], 'go', markersize=10, label='Начало',
             markerfacecolor='white')
    ax1.plot(x_opt[0], x_opt[1], 'r*', markersize=15,
             markerfacecolor='white')

    # Глобальный минимум функции Матьяса (0, 0)
    ax1.plot(0, 0, 'y*', markersize=20,
             label='Глобальный минимум (0, 0)', markerfacecolor='white', markeredgewidth=2)

    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('y', fontsize=12)
    ax1.set_title(f'Траектория Adagrad\nη={eta}, Итераций: {len(history_x) - 1}',
                  fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')  # КВАДРАТНЫЙ ГРАФИК - равный масштаб по осям
    plt.colorbar(contour, ax=ax1, label='F(x,y)')

    # 2. График сходимости значений функции
    f_values = [f(x) for x in history_x]
    ax2.semilogy(f_values, 'b-', linewidth=2)
    ax2.set_xlabel('Итерация', fontsize=12)
    ax2.set_ylabel('F(x)', fontsize=12)
    ax2.set_title('Сходимость функции', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.fill_between(range(len(f_values)), f_values, alpha=0.3, color='blue')

    plt.tight_layout()
    plt.show()

def create_animation(f, history_x, x0, x_opt, eta):
    """Анимация приближения к минимуму"""
    fig, ax = plt.subplots(figsize=(10, 10))  # КВАДРАТНЫЙ размер

    # Определяем границы
    x_min = history_x[:, 0].min() - 1
    x_max = history_x[:, 0].max() + 1
    y_min = history_x[:, 1].min() - 1
    y_max = history_x[:, 1].max() + 1

    # Делаем квадратные границы
    x_range = max(abs(x_min), abs(x_max))
    y_range = max(abs(y_min), abs(y_max))
    limit = max(x_range, y_range)
    x_min, x_max = -limit, limit
    y_min, y_max = -limit, limit

    x_range_vals = np.linspace(x_min, x_max, 150)
    y_range_vals = np.linspace(y_min, y_max, 150)
    X, Y = np.meshgrid(x_range_vals, y_range_vals)
    Z = np.array([[f(np.array([x, y])) for x in x_range_vals] for y in y_range_vals])

    # Фон - контурный график
    levels = np.linspace(Z.min(), Z.max(), 25)
    contour = ax.contourf(X, Y, Z, levels=levels, cmap='viridis', alpha=0.7)
    ax.contour(X, Y, Z, levels=levels, colors='black', alpha=0.3, linewidths=0.5)

    # Статические элементы
    ax.plot(0, 0, 'y*', markersize=25,
            markerfacecolor='white', markeredgewidth=2, label='Глобальный минимум (0, 0)')

    ax.plot(x0[0], x0[1], 'go', markersize=12, label='Начало',
            markerfacecolor='white', markeredgewidth=2)

    # Динамические элементы
    point, = ax.plot([], [], 'ro', markersize=10, label='Текущая точка',
                     markeredgewidth=2)
    line, = ax.plot([], [], 'r-', linewidth=2, alpha=0.8, label='Путь')

    # Информационный текст
    iteration_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                             fontsize=12, verticalalignment='top',
                             bbox=dict(boxstyle="round", facecolor="white", alpha=0.9))

    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title(f'Adagrad на функции Матьяса (η={eta})',
                 fontsize=16, fontweight='bold')

    # Легенда
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', edgecolor='black', label='Начальная точка'),
        Patch(facecolor='red', edgecolor='black', label='Текущая точка'),
        Patch(facecolor='red', edgecolor='black', label='Траектория', alpha=0.8),
        Patch(facecolor='yellow', edgecolor='black', label='Глобальный минимум'),
    ]
    ax.legend(handles=legend_elements, loc='lower left', fontsize=10)

    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')  # КВАДРАТНЫЙ ГРАФИК
    plt.colorbar(contour, ax=ax, label='F(x,y)', shrink=0.8)

    def init():
        point.set_data([], [])
        line.set_data([], [])
        iteration_text.set_text('')
        return point, line, iteration_text

    def animate(i):
        idx = min(i, len(history_x) - 1)
        point.set_data([history_x[idx, 0]], [history_x[idx, 1]])
        line.set_data(history_x[:idx + 1, 0], history_x[:idx + 1, 1])

        f_val = f(history_x[idx])
        iteration_text.set_text(f'Итерация: {idx}\nF(x) = {f_val:.4f}\n'
                                f'x = [{history_x[idx, 0]:.3f}, {history_x[idx, 1]:.3f}]')

        return point, line, iteration_text

    frames = len(history_x)
    ani = animation.FuncAnimation(fig, animate, frames=frames,
                                  init_func=init, blit=True, interval=50, repeat=False)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("=" * 90)

    print("ADAGRAD С ВИЗУАЛИЗАЦИЕЙ")

    print("=" * 90)

    print("Функция: Матьяса F(x) = 0.26(x**2+y**2)-0.48xy")

    print("Глобальный минимум: (0, 0) со значением F(0, 0) = 0")

    print("=" * 90)

    x0 = np.array([10.0, 10.0])

    η = 0.5

    epsilon = 1e-5

    print(f"Начальная точка: [{x0[0]}, {x0[1]}]")

    print(f"Learning rate: η = {η}")

    print(f"Критерий остановки: ε = {epsilon}")

    x_opt, iter_count, lr, G, history_x = adagrad_detailed(matyas_function, x0, epsilon=epsilon, eta=η)

    plot_trajectory(matyas_function, history_x, x0, x_opt, η)

    create_animation(matyas_function, history_x, x0, x_opt, η)
