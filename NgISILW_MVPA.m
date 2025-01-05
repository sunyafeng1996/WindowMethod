classdef NgISILW_MVPA < ALGORITHM
    % <single> <real/integer> <large/none> <constrained/none>
    % Most valuable player algorithm

    %------------------------------- Reference --------------------------------
    % H. Bouchekara, Most valuable player algorithm: a novel optimization
    % algorithm inspired from sport, Operational Research, 2020, 20: 139-195.
    %------------------------------- Copyright --------------------------------
    % Copyright (c) 2024 BIMK Group. You are free to use the PlatEMO for
    % research purposes. All publications which use this platform or any code
    % in the platform should acknowledge the use of "PlatEMO" and reference "Ye
    % Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
    % for evolutionary multi-objective optimization [educational forum], IEEE
    % Computational Intelligence Magazine, 2017, 12(4): 73-87".
    %--------------------------------------------------------------------------

    % This function is written by H. R. E. H. Bouchekara (email: bouchekara.houssem@gmail.com)

    methods
        function main(Algorithm,Problem)
            %% Parameter setting
            [TeamsSize,ArchiveSize,c1,c2] = Algorithm.ParameterSet(round(Problem.N/5),round(Problem.N/3),1,2);

            %% Generate random population
            Players = Problem.Initialization();
            [MVP,TEAMS,OBJS_TEAM,FranchisePlayer,npt] = InitializeMVPA(Problem,Players,TeamsSize);

            %% window hyperparameter
            t=40; wl_min=10;
            M=size(Players.objs,2);
            if size(Players.cons,2)==1
                if sum(Players.cons)==0
                    C=0;
                else
                    C=1;
                end
            else
                C=size(Players.cons,2);
            end
            ngisilw=NgISILW(Problem.D,wl_min,Problem.N,Problem.D,M,C,Players,t);

            gen=1;
            Algorithm.metric.conv(gen,1)=Players.best.objs;
            Algorithm.metric.bextX=Players.best.decs;
            Algorithm.metric.Min_value=Players.best.objs;

            %% Optimization
            while Algorithm.NotTerminated(Players)
                [~,rank]         = sort(Players.objs);
                Archive          = Players(rank(1:ArchiveSize));
                
                PlayersDec         = Players.decs;
                MVPDec             = MVP.decs;
                FranchisePlayerDec = FranchisePlayer.decs;
                [~,PROBLEMSIZE]    = size(PlayersDec);
                TeamsSize          = size(TEAMS,1);
                %% MVPA optimization
                for i = 1 : TeamsSize
                    % Selecting the first team
                    TEAMi = nonzeros(TEAMS(i,:));
                    % Selecting the second team
                    j = randi(TeamsSize); while j==i, j=randi(TeamsSize); end
                    % Competition phase
                    % Update of players skills of TEAMi
                    PlayersDec(TEAMi,:) = PlayersDec(TEAMi,:)+...
                        (c1*rand(npt(i),PROBLEMSIZE)).*(repmat(MVPDec,npt(i),1)-PlayersDec(TEAMi,:))+...
                        (c2*rand(npt(i),PROBLEMSIZE)).*(repmat(FranchisePlayerDec(i,:),npt(i),1)-PlayersDec(TEAMi,:));

                    TEAMWin = WinOrLoss(OBJS_TEAM,i,j); % check which team is winning

                    if TEAMWin == 1 % TEAMi beats TEAMj
                        PlayersDec(TEAMi,:) = PlayersDec(TEAMi,:)+rand*(PlayersDec(TEAMi,:)-repmat(FranchisePlayerDec(j,:),npt(i),1));
                    else % TEAMj beats TEAMi
                        PlayersDec(TEAMi,:) = PlayersDec(TEAMi,:)+rand*(repmat(FranchisePlayerDec(j,:),npt(i),1)-PlayersDec(TEAMi,:));
                    end
                    PlayersDec = bound(Problem,PlayersDec);
                end

                %% operator now dim
                offDec_window=Players.decs;
                for i =1:Problem.N
                    window=ngisilw.windows{i};
                    offDec_window(i,window)=PlayersDec(i,window);
                end
                Players_NEW = Problem.Evaluation(offDec_window);

                replace          = find(Players_NEW.objs<=Players.objs);
                Players(replace) = Players_NEW(replace);
                [MVP,TEAMS,OBJS_TEAM,FranchisePlayer]=ArgminMVPA(Players,TEAMS);
                Players_All      = [Archive, Players];
                [~,rank1]        = sort(Players_All);
                Players          = Players_All(rank1);
                
                %% update windows
                ngisilw=ngisilw.next_step(Players,gen);

                gen=gen+1;
                Algorithm.metric.conv(gen,1)=Players.best.objs;
                Algorithm.metric.bextX=Players.best.decs;
                Algorithm.metric.Min_value=Players.best.objs;
            end
        end
    end
end

function TEAMWin = WinOrLoss(Fitness,i,j)
    FitnessN = ((Fitness)-min(Fitness));
    p = 1-[FitnessN(i).^1/(sum(FitnessN(i).^1+FitnessN(j).^1)) FitnessN(j).^1/(sum(FitnessN(i).^1+FitnessN(j).^1))];
    if p(1) == p(2)
        if rand > 0.5
            TEAMWin = 1;
        else
            TEAMWin = 2;
        end
    else
        [p,ip]  = sort(p,'descend');
        TEAMWin = ip(1);
        if rand > p(1)
            TEAMWin = ip(2);
        end
    end
end

function x = bound(Problem,x)
    x = min(max(x,Problem.lower),Problem.upper);
end