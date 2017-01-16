#ifndef WINZENT_SIMULATION_CLASSREGISTRY_H
#define WINZENT_SIMULATION_CLASSREGISTRY_H


#include <assert>
#include <string>
#include <unordered_map>

#include <boost/function.hpp>
#include <boost/functional/factory.hpp>


namespace Winzent {


    template <class C>
    struct ClassRegistration
    {
        enum { Registered = 0 };
    };


    /*!
     * \brief A simple class registry for object serialization
     *
     * The ClassRegistry is a Singleton class that helps with registering
     * and dynamically creating classes. A class that wants to use this
     * facility needs to call the WINZENT_REGISTER_CLASS() macro in its
     * header file. This makes a struct `ClassRegistration<T>` available,
     * which serves as data object.
     *
     * Creating a new class can be done in two ways: First, by using
     * ClassRegistration<T>::createNew(), which works at compile time.
     * If the name of the class is only known at runtime,
     * ClassRegistry::metaObject() gives access to the registered class's
     * QMetaObject, which offers a QMetaObject::newInstance() method.
     *
     * To make this work, the public default constructor of the target
     * class must be declared with the `Q_INVOKABLE` macro.
     */
    template <class BaseClass>
    class ClassRegistry
    {
    public:


        /*!
         * \brief The factory method for the default constructor of
         *  `BaseClass`.
         */
        typedef boost::function<BaseClass* ()> Factory;


        /*!
         * \brief Returns the ClassRegistry's singleton instance
         *
         * \return The usable ClassRegistry instance
         */
        static ClassRegistry<BaseClass>* instance()
        {
            if (nullptr == ClassRegistry<BaseClass>::m_instance) {
                ClassRegistry<BaseClass>::m_instance =
                        new ClassRegistry<BaseClass>();
            }

            return ClassRegistry<BaseClass>::m_instance;
        }


        /*!
         * \brief Checks whether a given key is registered
         *
         * \param key The key to check for
         *
         * \return true if the key is registered, false if not.
         */
        bool isRegistered(std::string const& key) const
        {
            return m_registry.find(key) != m_registry.end();
        }


        /*!
         * \brief Registers a class
         *
         * \param key The class' meta object
         *
         * \return The internal ID
         */
        void registerClass(std::string const& key, Factory const& factory)
        {
            m_registry[key] = factory;
        }


        /*!
         * \brief Creates a new instance of the registered class
         *
         * \param[in] key The class' name
         *
         * \return The new object instance, as its base class
         */
        BaseClass* create(std::string const& key)
        {
            assert(isRegistered(key));
            return m_registry[key]();
        }


        template <class Derived>
        Derived* create(std::string const& key)
        {
            return reinterpret_cast<Derived*>(create(key));
        }


    private:


        //! Singleton instance
        static ClassRegistry<BaseClass>* m_instance;


        //! \brief The actual name => class mapping registry
        std::unordered_map<std::string, Factory> m_registry;


        //! Private default constructor
        explicit ClassRegistry()
        {
        }
    };


    template <class BaseClass>
    ClassRegistry<BaseClass>* ClassRegistry<BaseClass>::m_instance = nullptr;
} // namespace Winzent


#define WINZENT_REGISTER_CLASS(KLASS, BaseClass)                            \
namespace Winzent {                                                         \
    template <>                                                             \
    struct ClassRegistration<KLASS>                                         \
    {                                                                       \
        enum { Defined = 1 };                                               \
        static const int id;                                                \
                                                                            \
        static const char* className()                                      \
        {                                                                   \
            return #KLASS;                                                  \
        }                                                                   \
    };                                                                      \
                                                                            \
    const int ClassRegistration<KLASS>::id =                                \
            ClassRegistry<BaseClass>::instance()->registerClass(            \
                #KLASS,                                                     \
                []() { return new KLASS (); });                             \
}

#endif // WINZENT_SIMULATION_CLASSREGISTRY_H
