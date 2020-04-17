package GeneticTSP;


/**
 * 主函数运行类
 */

public class MainRun {

    public static void main(String[] args) {
        // TODO Auto-generated method stub
		long start,end;
		start = System.currentTimeMillis();

        //创建遗传算法驱动对象
        GeneticAlgorithm GA = new GeneticAlgorithm();

        //创建初始种群
        SpeciesPopulation speciesPopulation = new SpeciesPopulation();

        //开始遗传算法（选择算子、交叉算子、变异算子）
        SpeciesIndividual bestRate = GA.run(speciesPopulation);

        //打印路径与最短距离
        bestRate.printRate();


		end = System.currentTimeMillis();
		System.out.println("\nstart time:" + start+ "; end time:" + end+ "; Run Time:" + (end - start) + "(ms)");
    }

}
