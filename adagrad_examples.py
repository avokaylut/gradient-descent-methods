import numpy as np

import matplotlib.pyplot as plt

def compute_gradient(f, x, h=1e-7):
    """
    Вычисление градиента функции F в точке x
    ∇F(x) = [∂F/∂x₁, ∂F/∂x₂, ..., ∂F/∂xₙ]ᵀ
    """
    n = len(x)
    grad = np.zeros(n)

    for i in range(n):
        e_i = np.zeros(n)
        e_i[i] = 1.0
        x_plus = x + h * e_i
        x_minus = x - h * e_i
        grad[i] = (f(x_plus) - f(x_minus)) / (2 * h)
# Используется центральная разностная производная второго порядка точности
    return grad

def adagrad(f, x0, epsilon=1e-7, eta=0.1):
    """
    Метод Adagrad

    Параметры:
    f - минимизируемая функция F(x)
    x0 - начальное приближение x⁰
    epsilon - точность расчета ε
    eta - глобальная скорость обучения η
    """
    # Шаг 1: Инициализация
    x_j = np.asarray(x0, dtype=float).copy()
    j = 0

    # G - накопитель сумм квадратов градиентов( G только растет, что приводит к монотонному уменьшению шага.)
    G = np.zeros_like(x_j) # G_0 = 0
    epsilon_ada = 1e-8  # малая константа для численной стабильности

    # История для возврата
    history_x = [x_j.copy()]
    history_f = [f(x_j)]

    print(f"Adagrad: Начальное приближение: x⁰ = {x0}")
    print(f"Adagrad: F(x⁰) = {f(x0):.8f}")

    while True:
        if j >= 100000:
            print("Достигнут предел 100000 итераций")
            break
        # Шаг 2: Вычисление градиента
        grad_F = compute_gradient(f, x_j)

        # Шаг 3: Обновление аккумулятора G
        # G_{j,i} = G_{j-1,i} + (g_{j,i})²
        G = G + grad_F ** 2

        # Шаг 4: Вычисление адаптивной скорости обучения (Чем больше G (много обновлений) → меньше шаг, и наоборот)
        # adaptive_lr_i = η / (√(G_{j,i}) + ε)
        adaptive_lr = eta / (np.sqrt(G) + epsilon_ada)

        # Шаг 5: Обновление параметров
        # x_{j+1,i} = x_{j,i} - adaptive_lr_i * g_{j,i} матричная форма уравнения обновления
        x_j_plus_1 = x_j - adaptive_lr * grad_F

        if not np.isfinite(x_j_plus_1).all():
            raise ValueError("Расходящиеся итерации: уменьшите eta")
        history_x.append(x_j_plus_1.copy())
        history_f.append(f(x_j_plus_1))

        # Шаг 6: Проверка условий остановки
        diff_x = np.linalg.norm(x_j_plus_1 - x_j)  # |x^{j+1} - x^j|
        diff_f = abs(f(x_j_plus_1) - f(x_j))  # |F(x^{j+1}) - F(x^j)|

        # print(f"Итерация {j}: x = {x_j_plus_1}, F(x) = {f(x_j_plus_1):.8f}")

        # Проверяем условия остановки
        if diff_x < epsilon: # погрешность вычисления координат точки минимума
            print(f"Остановка по изменению x: |Δx| = {diff_x:.2e} < ε")
            break
        elif diff_f < epsilon: # отн. погрешность вычисления мин зн-я целевой функции
            print(f"Остановка по изменению F(x): |ΔF| = {diff_f:.2e} < ε")
            break

        # Переход к следующей итерации
        x_j = x_j_plus_1
        j += 1

    print(f"\nРезультат Adagrad:")
    print(f"Найденный минимум: x* = {x_j_plus_1}")
    print(f"Значение функции: F(x*) = {f(x_j_plus_1):.10f}")
    print(f"Количество итераций: {len(history_x) - 1}")
    # print(f"Количество точек в истории: {len(history_x)}")

    return x_j_plus_1, history_x, history_f

def test_rosenbrock():
    """Функция Розенброка"""
    print("=" * 60)
    print("ТЕСТ 1: Функция Розенброка")
    print("F(x, y) = (1 - x)² + 100(y - x²)²")
    print("Минимум в точке (1, 1), F(x*) = 0")
    print("=" * 60)

    def F(x):
        return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2

    x0 = np.array([0.0, 0.0])
    epsilon = 1e-7

    x_opt, history_x, history_f = adagrad(F, x0, epsilon, eta=0.02)

    # Визуализация
    x_vals = np.linspace(-2, 2, 100)
    y_vals = np.linspace(-1, 3, 100)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = (1 - X) ** 2 + 100 * (Y - X ** 2) ** 2

    plt.figure(figsize=(15, 5))

    ax1 = plt.subplot(1, 3, 1, projection='3d')
    ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)
    ax1.plot([h[0] for h in history_x], [h[1] for h in history_x], history_f,
             'ro-', linewidth=2, markersize=4)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('F(x, y)')
    ax1.set_title('3D визуализация')

    plt.subplot(1, 3, 2)
    plt.contour(X, Y, Z, levels=50)
    plt.plot([h[0] for h in history_x], [h[1] for h in history_x], 'ro-',
             linewidth=2, markersize=4)
    plt.plot(1, 1, 'g*', markersize=15, label='Минимум (1, 1)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Линии уровня')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.plot(history_f, 'b-o', linewidth=2, markersize=3)
    plt.yscale('log')
    plt.xlabel('Итерация')
    plt.ylabel('F(x) (лог. шкала)')
    plt.title('Сходимость')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def test_rastrigin():
    """Функция Растригина"""
    print("\n" + "=" * 60)
    print("ТЕСТ 2: Функция Растригина")
    print("F(x, y) = 20 + (x² - 10cos(2πx)) + (y² - 10cos(2πy))")
    print("Минимум в точке (0, 0), F(x*) = 0")
    print("=" * 60)

    def F(x):
        return 20 + (x[0] ** 2 - 10 * np.cos(2 * np.pi * x[0])) + (x[1] ** 2 - 10 * np.cos(2 * np.pi * x[1]))

    x0 = np.array([2.0, 2.0])
    epsilon = 1e-7

    x_opt, history_x, history_f = adagrad(F, x0, epsilon, eta=0.1)

    # Визуализация
    x_vals = np.linspace(-3, 3, 100)
    y_vals = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = 20 + (X ** 2 - 10 * np.cos(2 * np.pi * X)) + (Y ** 2 - 10 * np.cos(2 * np.pi * Y))

    plt.figure(figsize=(15, 5))

    ax1 = plt.subplot(1, 3, 1, projection='3d')
    ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)
    ax1.plot([h[0] for h in history_x], [h[1] for h in history_x], history_f,
             'ro-', linewidth=2, markersize=4)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('F(x, y)')
    ax1.set_title('3D визуализация')

    plt.subplot(1, 3, 2)
    plt.contour(X, Y, Z, levels=20)
    plt.plot([h[0] for h in history_x], [h[1] for h in history_x], 'ro-',
             linewidth=2, markersize=4)
    plt.plot(0, 0, 'g*', markersize=15, label='Минимум (0, 0)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Линии уровня')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.plot(history_f, 'b-o', linewidth=2, markersize=3)
    plt.yscale('log')
    plt.xlabel('Итерация')
    plt.ylabel('F(x) (лог. шкала)')
    plt.title('Сходимость')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def test_himmelblau():
    """Функция Химмельблау"""
    print("\n" + "=" * 60)
    print("ТЕСТ 3: Функция Химмельблау")
    print("F(x, y) = (x² + y - 11)² + (x + y² - 7)²")
    print("4 минимума: (3, 2), (-2.8, 3.1), (-3.8, -3.3), (3.6, -1.8)")
    print("=" * 60)

    def F(x):
        return (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2

    x0 = np.array([0.0, 0.0])
    epsilon = 1e-7

    x_opt, history_x, history_f = adagrad(F, x0, epsilon, eta=0.5)

    # Визуализация
    x_vals = np.linspace(-5, 5, 100)
    y_vals = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = (X ** 2 + Y - 11) ** 2 + (X + Y ** 2 - 7) ** 2

    plt.figure(figsize=(15, 5))

    ax1 = plt.subplot(1, 3, 1, projection='3d')
    ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)
    ax1.plot([h[0] for h in history_x], [h[1] for h in history_x], history_f,
             'ro-', linewidth=2, markersize=4)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('F(x, y)')
    ax1.set_title('3D визуализация')

    plt.subplot(1, 3, 2)
    plt.contour(X, Y, Z, levels=50)
    plt.plot([h[0] for h in history_x], [h[1] for h in history_x], 'ro-',
             linewidth=2, markersize=4)
    minima = [(3, 2), (-2.8, 3.1), (-3.8, -3.3), (3.6, -1.8)]
    for min_point in minima:
        plt.plot(min_point[0], min_point[1], 'g*', markersize=10)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Линии уровня')
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.plot(history_f, 'b-o', linewidth=2, markersize=3)
    plt.yscale('log')
    plt.xlabel('Итерация')
    plt.ylabel('F(x) (лог. шкала)')
    plt.title('Сходимость')
    plt.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    test_rosenbrock()
    test_rastrigin()
    test_himmelblau()
