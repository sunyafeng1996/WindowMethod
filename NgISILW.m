classdef NgISILW
    % 神经网络指导的维度序列 Neural network-guided dimension sequence
    % 基于等距分段的窗口长度 Isometric segmentation-based window length

    properties
        wl_max
        wl_min
        N
        D
        windows
        %% 窗口长度等距分段
        wls
        ds % 维度序列 Dimension Sequence
        dr % 维度范围
        %% 神经网络结构
        M % 目标数量
        C % 约束数量
        net
        %% 神经网络训练
        lr %学习率
        momentum %动量
        numEpochs; % 训练轮次
        t; %每隔t代训练一次神经网络
    end

    methods
        function obj = NgISILW(wl_max,wl_min,N,D,M,C,Population,t)
            obj.wl_max = wl_max;
            obj.wl_min = wl_min;
            obj.N = N;
            obj.D = D;
            %% 神经网络参数
            obj.M=M;
            obj.C=C;
            obj.lr=0.001;
            obj.momentum=0.95;
            obj.numEpochs=100;
            obj.t=t;
            %% 全连接网络
            obj=obj.create_network();
            obj=obj.train_network(Population);
            %% 维度序列
            obj=obj.dim_sorted();
            %% 维度范围
            temp=round((obj.D/(obj.N+1)):(obj.D/(obj.N+1)):obj.D);
            if temp(end)~=obj.D, temp(end)=obj.D; end
            obj.dr(1,:)=temp(1:obj.N);
            obj.dr(2,:)=temp(2:end);
            %% 生成均匀尺寸的窗口
            obj.wls=round(rand(1,obj.N).*(obj.dr(2,:)-obj.dr(1,:))+obj.dr(1,:));
            obj.wls(obj.wls>obj.D)=obj.D;
            obj.wls(obj.wls<obj.wl_min)=obj.wl_min;
            r1=randperm(obj.N); r2=randperm(obj.N); 
            for i=1:obj.N
                obj.windows{i}=obj.ds(r1(i),1:obj.wls(r2(i)));
            end
        end

        % %% 初始化网络
        % function obj = create_network(obj)
        %     layers = [featureInputLayer(obj.D)
        %         fullyConnectedLayer(obj.N)
        %         fullyConnectedLayer(obj.M+obj.C)];
        %     obj.net=dlnetwork(layers);
        % end

        %% 初始化网络
        function obj = create_network(obj)
            layers = [imageInputLayer([obj.D 1 1], 'Normalization', 'none', 'Name', 'input')
                fullyConnectedLayer(obj.N, 'Name', 'fc1')
                fullyConnectedLayer(obj.M+obj.C, 'Name', 'fc2')];
            lgraph = layerGraph(layers);
		  obj.net = dlnetwork(lgraph);
        end

        %% 全连接训练
        function obj = train_network(obj,Population)
            dst=DataSet(Population,obj.C);
            vel = [];
            epoch = 0;
            while epoch < obj.numEpochs
                epoch = epoch + 1;
                [~,gradients] = dlfeval(@modelLoss,obj.net,dst.samples_normalized_dl,dst.labels_normalized_dl);
                [obj.net,vel] = sgdmupdate(obj.net,gradients,vel,obj.lr,obj.momentum);
            end
        end

        %% 获取维度排序
        function obj = dim_sorted(obj)
            sorces=abs(extractdata(obj.net.Learnables(1,:).Value{1}));
            [~,I]=sort(sorces','descend');
            obj.ds=I';
        end

        %% 下一步
        function obj = next_step(obj,Population,gen)
            if mod(gen,obj.t)==0
                %% 全连接网络
                obj=obj.create_network();
                obj=obj.train_network(Population);
                %% 维度排序
                obj=obj.dim_sorted();
            end
            %% 生成等距分段内随机尺寸的窗口
            obj.wls=round(rand(1,obj.N).*(obj.dr(2,:)-obj.dr(1,:))+obj.dr(1,:));
            obj.wls(obj.wls>obj.D)=obj.D;
            obj.wls(obj.wls<obj.wl_min)=obj.wl_min;
            r1=randperm(obj.N); r2=randperm(obj.N); 
            for i=1:obj.N
                obj.windows{i}=obj.ds(r1(i),1:obj.wls(r2(i)));
            end
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end
end


function [loss,gradients] = modelLoss(net,X,T)
Y = forward(net,X);
% loss = l2loss(Y,T);
loss = mse(Y,T);
gradients = dlgradient(loss,net.Learnables);
end

