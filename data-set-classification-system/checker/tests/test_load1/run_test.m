function run_test()
    fout = fopen("out", "w+");
    [X, y] = load_dataset("../../input/ex1.mat");
    fdisp(fout, [X, y]);
    fclose(fout);
endfunction