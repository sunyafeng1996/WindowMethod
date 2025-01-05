classdef Window
    % 窗口大小向优秀的窗口大小的分布去学习
    % 个体有概率向优秀分布学习
    % f1 
    properties
        wl_max
        wl_min
        N
        D
        window
        wl
    end

    methods
        function obj = Window(wl_max,wl_min,N,D)
            obj.wl_max = wl_max;
            obj.wl_min = wl_min;
            obj.N = N;
            obj.D = D;
            obj.wl=randi(obj.wl_max-obj.wl_min)+obj.wl_min;
            ds=randperm(obj.D);
            obj.window=ds(1:obj.wl);
        end

        function obj = next_step(obj)
            obj.wl=randi(obj.wl_max-obj.wl_min)+obj.wl_min;
            ds=randperm(obj.D);
            obj.window=ds(1:obj.wl);
        end

    end
end

