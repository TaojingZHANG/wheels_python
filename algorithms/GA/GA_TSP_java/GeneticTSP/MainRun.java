package GeneticTSP;


/**
 * ������������
 */

public class MainRun {

    public static void main(String[] args) {
        // TODO Auto-generated method stub
		long start,end;
		start = System.currentTimeMillis();

        //�����Ŵ��㷨��������
        GeneticAlgorithm GA = new GeneticAlgorithm();

        //������ʼ��Ⱥ
        SpeciesPopulation speciesPopulation = new SpeciesPopulation();

        //��ʼ�Ŵ��㷨��ѡ�����ӡ��������ӡ��������ӣ�
        SpeciesIndividual bestRate = GA.run(speciesPopulation);

        //��ӡ·������̾���
        bestRate.printRate();


		end = System.currentTimeMillis();
		System.out.println("\nstart time:" + start+ "; end time:" + end+ "; Run Time:" + (end - start) + "(ms)");
    }

}
