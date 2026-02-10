# **TASK 1:**
Заполнить масив float/double значениями синуса (один период на всю длину массива), количество элементов $10^7$. Посчитать сумму и вывести в терминал.
## Сборка (запуск из директории с файлом исходного кода и CMakeLists.txt):
Для float: 
```bash
cmake -S . -B build_float -DUSE_DOUBLE=OFF ; cmake --build build_float
```
Для double (вариант по умолчанию):
```bash
cmake -S . -B build_double ; cmake --build build_double
```
## Результат работы:
С использованием float:
```bash
Summa of Sin values using type float:
7.5007e-05
```
С использованием double:
```bash
Summa of Sin values using type double:
-6.76916e-10
```