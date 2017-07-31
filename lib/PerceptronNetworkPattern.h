#ifndef WZANN_PERCEPTRONNETWORKPATTERN_H_
#define WZANN_PERCEPTRONNETWORKPATTERN_H_


#include "Vector.h"
#include "NeuralNetworkPattern.h"


namespace wzann {
    class NeuralNetwork;


    /*!
     * \brief Instances of this class represent the pattern to create a
     *  (potentially multi-layered), feed-forward perceptron without
     *  recurrent connections, shortcut connections or other specialities.
     */
    class PerceptronNetworkPattern : public NeuralNetworkPattern
    {
    public:


        /*!
         * \brief Creates a new, empty PerceptronNetworkPattern.
         *
         * \sa NeuralNetworkPattern#addLayer()
         */
        PerceptronNetworkPattern();


        virtual ~PerceptronNetworkPattern();


        /*!
         * \brief Creates a clone of a pattern instance.
         *
         * \sa NeuralNetworkPattern#clone
         */
        virtual NeuralNetworkPattern* clone() const override;


        /*!
         * \brief Checks for equality of two PerceptronNetworkPatterns
         *
         * \param[in] other The other pattern
         *
         * \return True if the two are of the same class and have the
         *  same parameters
         */
        virtual bool operator ==(NeuralNetworkPattern const& other)
                const override;


    protected:


        /*!
         * \brief Configures the supplied neural network
         *  to be an perceptron.
         */
        virtual void configureNetwork(NeuralNetwork& network) override;


        /*!
         * \brief Calculates the result of running an input vector through
         *  a Perceptron.
         *
         * \sa NeuralNetworkPattern#calculate
         */
        virtual Vector calculate(
                NeuralNetwork& network,
                Vector const& input)
                override;
    };
} // namespace wzann

#endif // WZANN_PERCEPTRONNETWORKPATTERN_H_
